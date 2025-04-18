"""
Simple Edge Device Client

Streams audio from microphone to server at 16kHz and plays audio from server at 24kHz.

Usage:
    python -m audio.client.simple_client [--server=ws://localhost:8765]
"""

import asyncio
import websockets
import json
import base64
import numpy as np
import pyaudio
import wave
import io
import time
import argparse

class SimpleAudioClient:
    def __init__(self, server_url="ws://localhost:8765"):
        """Initialize audio client"""
        self.server_url = server_url
        self.websocket = None
        self.pyaudio = None
        self.mic_stream = None
        self.speaker_stream = None
        self.running = False
        self.sequence = 0  # Counter for message sequencing
        
        # Audio configuration
        self.channels = 1  # Mono
        self.mic_sample_rate = 16000  # 16kHz for whisper
        self.speaker_sample_rate = 24000  # 24kHz for TTS audio
        self.chunk_size = 1024  # Samples per chunk
        
        # For handling chunked audio from server
        self.audio_chunks = {}  # Dictionary to store partial audio chunks
        
        # Try to import the signal module for resampling
        try:
            from scipy import signal
            self.has_signal = True
        except ImportError:
            print("scipy.signal not available, audio resampling will be limited")
            self.has_signal = False
        
    async def connect(self):
        """Connect to server and initialize audio devices"""
        try:
            # Connect to WebSocket server
            print(f"Connecting to {self.server_url}...")
            self.websocket = await websockets.connect(self.server_url)
            print("Connected to server")
            
            # Initialize PyAudio
            self.pyaudio = pyaudio.PyAudio()
            
            # Set up microphone input stream
            print("Initializing microphone...")
            self.mic_stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.mic_sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            # Set up speaker output stream
            print("Initializing speaker...")
            self.speaker_stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.speaker_sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
            print("Audio devices initialized")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            if self.websocket:
                await self.websocket.close()
            self.cleanup()
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
            self.mic_stream = None
            
        if self.speaker_stream:
            self.speaker_stream.stop_stream()
            self.speaker_stream.close()
            self.speaker_stream = None
            
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None
    
    async def disconnect(self):
        """Disconnect from server and clean up"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            
        self.cleanup()
        print("Disconnected from server")
        
    async def run(self):
        """Run the client"""
        if not await self.connect():
            return
            
        self.running = True
        
        print("\n========== EDGE DEVICE SIMULATOR ==========")
        print("Streaming microphone audio to server...")
        print("Receiving audio will play through speakers")
        print("\n⚠️  IMPORTANT: Use headphones to prevent echo!")
        print("==========================================")
        print("Press Ctrl+C to exit\n")
        
        # Start the send and receive loops
        try:
            send_task = asyncio.create_task(self.send_audio_loop())
            receive_task = asyncio.create_task(self.receive_audio_loop())
            await asyncio.gather(send_task, receive_task)
        except asyncio.CancelledError:
            print("Tasks cancelled")
        finally:
            self.running = False
            await self.disconnect()
            
    async def send_audio_loop(self):
        """Continuously read from microphone and send to server"""
        try:
            while self.running and self.websocket:
                # Read audio data from microphone
                try:
                    audio_data = self.mic_stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Simple audio level check
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    audio_level = np.abs(audio_np).mean()
                    
                    if audio_level < 10 and self.sequence % 100 == 0:
                        print(f"Warning: Very low audio level ({audio_level:.1f}). Check your microphone.")
                    
                    # Create audio message
                    message = {
                        "type": "audio_stream",
                        "timestamp": time.time(),
                        "sequence": self.sequence,
                        "payload": {
                            "audio": base64.b64encode(audio_data).decode('utf-8')
                        }
                    }
                    self.sequence += 1
                    
                    # Send to server
                    await self.websocket.send(json.dumps(message))
                    
                    # Pace ourselves according to the chunk duration
                    chunk_duration = self.chunk_size / self.mic_sample_rate
                    await asyncio.sleep(chunk_duration * 0.5)  # Sleep for half the chunk duration
                    
                except Exception as e:
                    print(f"Error capturing audio: {e}")
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            print(f"Error in send loop: {e}")
            self.running = False
            
    async def receive_audio_loop(self):
        """Receive audio and other messages from server"""
        try:
            while self.running and self.websocket:
                # Wait for a message from the server
                message = await self.websocket.recv()
                
                try:
                    # Parse the message
                    msg = json.loads(message)
                    
                    # Handle audio playback messages
                    if msg["type"] == "audio_playback":
                        # Check if this is a chunked message
                        chunk_info = msg["payload"].get("chunk_info", None)
                        
                        if chunk_info:
                            # This is a chunked message
                            chunk_index = chunk_info["index"]
                            total_chunks = chunk_info["total"]
                            is_wav = chunk_info.get("is_wav", False)
                            
                            # Generate a unique ID for this batch of chunks based on timestamp
                            batch_id = msg.get("timestamp", time.time())
                            
                            # Create entry in audio_chunks if it doesn't exist
                            if batch_id not in self.audio_chunks:
                                self.audio_chunks[batch_id] = {
                                    "chunks": [None] * total_chunks,
                                    "received": 0,
                                    "total": total_chunks,
                                    "is_wav": is_wav
                                }
                            
                            # Store this chunk
                            audio_data = base64.b64decode(msg["payload"]["audio"])
                            self.audio_chunks[batch_id]["chunks"][chunk_index] = audio_data
                            self.audio_chunks[batch_id]["received"] += 1
                            
                            # Check if we have all chunks
                            if self.audio_chunks[batch_id]["received"] == total_chunks:
                                # Process complete audio
                                if is_wav:
                                    # For WAV, each chunk has header. Extract only first header
                                    wav_header = self.audio_chunks[batch_id]["chunks"][0][:44]
                                    
                                    # Combine audio data (removing headers except for first chunk)
                                    combined_data = wav_header
                                    for i, chunk in enumerate(self.audio_chunks[batch_id]["chunks"]):
                                        if i == 0:
                                            combined_data += chunk[44:]
                                        else:
                                            combined_data += chunk[44:]
                                            
                                    # Process the complete WAV
                                    self.process_and_play_audio(combined_data)
                                else:
                                    # For non-WAV, just concatenate the chunks
                                    combined_data = b''.join(self.audio_chunks[batch_id]["chunks"])
                                    self.process_and_play_audio(combined_data)
                                    
                                # Remove this batch from the dictionary
                                del self.audio_chunks[batch_id]
                                
                                # Clean up old batches that might have been incomplete
                                current_time = time.time()
                                old_batches = [bid for bid in self.audio_chunks 
                                              if isinstance(bid, float) and current_time - bid > 30]
                                for old_id in old_batches:
                                    del self.audio_chunks[old_id]
                        else:
                            # Regular non-chunked audio message
                            audio_data = base64.b64decode(msg["payload"]["audio"])
                            self.process_and_play_audio(audio_data)
                            
                    # Handle status messages (transcription, state updates)
                    elif msg["type"] == "status":
                        if "transcription" in msg["payload"]:
                            text = msg["payload"]["transcription"]["text"]
                            print(f"🎤 Transcription: {text}")
                        elif "state" in msg["payload"]:
                            state = msg["payload"]["state"]
                            is_speech = msg["payload"].get("is_speech", False)
                            
                            if is_speech:
                                print("🔴 Recording...")
                            elif state == "processing":
                                print("⏱️ Processing...")
                            else:
                                print("⚪ Listening...")
                    
                    # Handle error messages
                    elif msg["type"] == "error":
                        print(f"❌ Error from server: {msg['payload']['error']}")
                        
                except json.JSONDecodeError:
                    print(f"Invalid JSON: {message[:100]}...")
                except Exception as e:
                    print(f"Error processing message: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except websockets.exceptions.ConnectionClosed:
            print("Connection to server closed")
        except Exception as e:
            print(f"Error in receive loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            
    def process_and_play_audio(self, audio_data):
        """Process audio data and play it"""
        try:
            # Check if it's a WAV file (has RIFF header)
            if audio_data[:4] == b'RIFF':
                # Parse WAV data
                with io.BytesIO(audio_data) as wav_buffer:
                    with wave.open(wav_buffer, 'rb') as wf:
                        # Get WAV properties
                        wav_rate = wf.getframerate()
                        wav_channels = wf.getnchannels()
                        
                        # Read audio data
                        pcm_data = wf.readframes(wf.getnframes())
                        
                        # Check if we need to resample
                        if wav_rate != self.speaker_sample_rate:
                            # Convert to numpy array for resampling
                            audio_np = np.frombuffer(pcm_data, dtype=np.int16)
                            
                            # Reshape for multiple channels if needed
                            if wav_channels > 1:
                                audio_np = audio_np.reshape(-1, wav_channels)
                                # Convert to mono by averaging channels
                                audio_np = np.mean(audio_np, axis=1, dtype=np.int16)
                            
                            # Resample to speaker rate if scipy is available
                            if self.has_signal:
                                from scipy import signal
                                resampled = signal.resample_poly(
                                    audio_np,
                                    self.speaker_sample_rate,
                                    wav_rate
                                )
                                
                                # Convert back to int16
                                pcm_data = np.int16(resampled).tobytes()
                            else:
                                # Simple DIY resampling if scipy not available
                                # This is not high quality but better than nothing
                                ratio = self.speaker_sample_rate / wav_rate
                                if ratio > 1:
                                    # Upsample by repeating samples
                                    indices = np.floor(np.arange(0, len(audio_np)) / ratio).astype(int)
                                    pcm_data = np.int16(audio_np[indices]).tobytes()
                                else:
                                    # Downsample by skipping samples
                                    indices = np.floor(np.arange(0, len(audio_np) * ratio) / ratio).astype(int)
                                    pcm_data = np.int16(audio_np[indices]).tobytes()
            else:
                # Assume it's raw PCM at the speaker sample rate
                pcm_data = audio_data
                
            # Play the audio
            self.speaker_stream.write(pcm_data)
            
        except Exception as audio_err:
            print(f"Error processing audio: {audio_err}")
            import traceback
            traceback.print_exc()

async def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simple Audio Edge Client")
    parser.add_argument("--server", default="ws://localhost:8765", help="WebSocket server URL")
    args = parser.parse_args()
    
    # Create and run client
    client = SimpleAudioClient(server_url=args.server)
    
    try:
        await client.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if client.running:
            await client.disconnect()
    
if __name__ == "__main__":
    asyncio.run(main()) 