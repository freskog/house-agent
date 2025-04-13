"""
Simple Edge Device Client

Streams audio from microphone to server at 16kHz and plays audio from server at 24kHz.

Usage:
    python -m audio.client.simple_client [--server=ws://localhost:8765]
"""

import asyncio
import websockets
import base64
import json
import time
import pyaudio
import numpy as np
import wave
import io
import argparse
from scipy import signal

class SimpleAudioClient:
    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.sequence = 0
        
        # Audio configuration
        self.mic_sample_rate = 16000         # 16kHz for Whisper
        self.speaker_sample_rate = 24000     # 24kHz for Kokoro TTS
        self.chunk_size = 1024               # Process audio in these sized chunks
        self.channels = 1                    # Mono audio
        
        # PyAudio objects
        self.pyaudio = None
        self.mic_stream = None
        self.speaker_stream = None
        
        # Control flags
        self.running = False
        
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
                        # Decode audio data
                        audio_data = base64.b64decode(msg["payload"]["audio"])
                        
                        # Process the audio (it could be a WAV file or raw PCM)
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
                                            
                                            # Resample to speaker rate
                                            resampled = signal.resample_poly(
                                                audio_np,
                                                self.speaker_sample_rate,
                                                wav_rate
                                            )
                                            
                                            # Convert back to int16
                                            pcm_data = np.int16(resampled).tobytes()
                            else:
                                # Assume it's raw PCM at the speaker sample rate
                                pcm_data = audio_data
                                
                            # Play the audio
                            self.speaker_stream.write(pcm_data)
                            
                        except Exception as audio_err:
                            print(f"Error processing audio: {audio_err}")
                            
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
                    
        except websockets.exceptions.ConnectionClosed:
            print("Connection to server closed")
        except Exception as e:
            print(f"Error in receive loop: {e}")
        finally:
            self.running = False
            
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