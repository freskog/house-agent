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
import queue

class SimpleAudioClient:
    def __init__(self, server_url="ws://localhost:8765"):
        """Initialize audio client"""
        self.server_url = server_url
        self.websocket = None
        self.pyaudio = None
        self.mic_stream = None
        self.speaker_stream = None
        self.running = False
        self.hanging_up = False  # Track when we're in the process of hanging up
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
        print("Cleaning up audio resources...")
            
        if self.mic_stream:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except Exception as e:
                print(f"Error closing mic stream: {e}")
            finally:
                self.mic_stream = None
            
        if self.speaker_stream:
            try:
                self.speaker_stream.stop_stream()
                self.speaker_stream.close()
            except Exception as e:
                print(f"Error closing speaker stream: {e}")
            finally:
                self.speaker_stream = None
            
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
            finally:
                self.pyaudio = None
        
        print("Audio resources cleaned up")
    
    async def disconnect(self):
        """Disconnect from server and clean up"""
        print("Disconnecting from server...")
        
        try:
            if self.websocket and not (hasattr(self.websocket, 'closed') and self.websocket.closed):
                # Send disconnect message
                try:
                    message = {
                        "type": "disconnect",
                        "timestamp": time.time(),
                        "sequence": self.sequence,
                        "payload": {
                            "reason": "Client disconnecting"
                        }
                    }
                    self.sequence += 1
                    await self.websocket.send(json.dumps(message))
                    print("Sent disconnect message")
                except Exception as e:
                    print(f"Error sending disconnect message: {e}")
                
                # Close the websocket
                try:
                    await self.websocket.close()
                except Exception as e:
                    print(f"Error closing websocket: {e}")
        except Exception as e:
            print(f"Error during disconnect: {e}")
        finally:
            self.websocket = None
            
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
            
            # Wait for both tasks to complete or until self.running becomes False
            while self.running:
                # Check if either task has completed
                if send_task.done() or receive_task.done():
                    break
                    
                # Short sleep to prevent high CPU usage
                await asyncio.sleep(0.1)
                
            # Once we've exited the loop (either because self.running is False or a task completed),
            # cancel any remaining tasks
            if not send_task.done():
                send_task.cancel()
            if not receive_task.done():
                receive_task.cancel()
                
            # Try to wait for the tasks to be properly cancelled
            try:
                await asyncio.wait_for(asyncio.gather(send_task, receive_task, return_exceptions=True), timeout=2.0)
            except asyncio.TimeoutError:
                print("Some tasks did not terminate gracefully")
                
        except asyncio.CancelledError:
            print("Tasks cancelled")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Client shutting down...")
            self.running = False
            await self.disconnect()
            self.cleanup()
            print("Client shutdown complete")
            
    async def send_audio_loop(self):
        """Continuously read from microphone and send to server"""
        try:
            while self.running and self.websocket:
                # If we're hanging up, stop sending audio
                if self.hanging_up:
                    print("Hanging up - stopped sending audio")
                    self.running = False  # Set running to false to trigger client shutdown
                    break
                    
                # Check if the websocket is closed
                if hasattr(self.websocket, 'closed') and self.websocket.closed:
                    print("WebSocket connection closed - stopped sending audio")
                    break
                    
                # Read audio data from microphone
                try:
                    audio_data = self.mic_stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Simple audio level check
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    audio_level = np.abs(audio_np).mean()
                    
                    # Only warn if audio level is very low and not too frequently
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
                    
                    # Send to server - wrap in try/except to handle closed connection
                    try:
                        await self.websocket.send(json.dumps(message))
                    except websockets.exceptions.ConnectionClosed:
                        print("Connection closed while sending audio - stopping")
                        self.running = False
                        break
                    
                    # Pace ourselves according to the chunk duration
                    chunk_duration = self.chunk_size / self.mic_sample_rate
                    await asyncio.sleep(chunk_duration * 0.5)  # Sleep for half the chunk duration
                    
                except Exception as e:
                    print(f"Error capturing audio: {e}")
                    await asyncio.sleep(0.1)
                    
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed - audio sending stopped")
            self.running = False
        except Exception as e:
            print(f"Error in send loop: {e}")
            self.running = False
            
    async def receive_audio_loop(self):
        """Receive audio and other messages from server"""
        try:
            while self.running and self.websocket:
                # Check if the websocket is closed
                if hasattr(self.websocket, 'closed') and self.websocket.closed:
                    print("WebSocket connection closed - stopped receiving")
                    self.running = False
                    break
                
                # Wait for a message from the server with a timeout
                try:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    # No message received in the timeout period, check if we're still running and try again
                    if not self.running or (hasattr(self.websocket, 'closed') and self.websocket.closed):
                        break
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed while waiting for messages")
                    self.running = False
                    break
                
                try:
                    # Parse the message
                    msg = json.loads(message)
                    
                    # Handle audio playback messages
                    if msg["type"] == "audio_playback":
                        # Check if this message has the hangup flag
                        is_hangup = msg["payload"].get("is_hangup", False)
                        is_final = msg["payload"].get("is_final", False)
                        
                        # If this is a hangup message, mark it for proper handling
                        if is_hangup:
                            print("📞 Received audio with hangup flag")
                            self.hanging_up = True
                            
                            # If this is a final empty message with hangup flag, terminate immediately
                            if is_final and "audio" not in msg["payload"]:
                                print("📞 Received final hangup signal without audio, terminating client...")
                                self.running = False
                                break
                        
                        # Check if there's audio data to play
                        if "audio" in msg["payload"]:
                            # Decode audio data from base64
                            audio_data = base64.b64decode(msg["payload"]["audio"])
                            
                            # Play it directly - no queuing
                            if audio_data and len(audio_data) > 0:
                                print(f"Playing audio: hangup={is_hangup}, final={is_final}, size={len(audio_data)}bytes")
                                self.process_and_play_audio(audio_data)
                                
                                # Check if this was the final audio chunk in a hangup sequence
                                if is_hangup and is_final:
                                    print("📞 Received final hangup audio, client will terminate after playback")
                            else:
                                print("Received empty audio data")
                    
                    # Handle status messages (transcription, state updates)
                    elif msg["type"] == "status":
                        # Check if we're hanging up
                        if msg["payload"].get("state") == "hanging_up":
                            print("Server is hanging up")
                            self.hanging_up = True
                            
                        # Handle speech detection
                        if "is_speech" in msg["payload"]:
                            is_speech = msg["payload"]["is_speech"]
                            if is_speech:
                                print("🎤 Speech detected...")
                                
                        # Handle state transitions
                        if "state" in msg["payload"]:
                            state = msg["payload"]["state"]
                            print(f"🔄 State: {state}")
                            
                        # Handle transcription
                        if "transcription" in msg["payload"]:
                            text = msg["payload"]["transcription"]["text"]
                            print(f"🔊 You said: \"{text}\"")
                    
                    # Handle error messages
                    elif msg["type"] == "error":
                        error = msg["payload"].get("error", "Unknown error")
                        print(f"❌ Error: {error}")
                
                except json.JSONDecodeError:
                    print(f"Invalid JSON message")
                except KeyError as e:
                    print(f"Missing key in message: {e}")
                except Exception as e:
                    print(f"Error processing message: {e}")
                    import traceback
                    traceback.print_exc()
                
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
            
            # Play the audio directly
            if self.speaker_stream:
                print(f"▶️ Playing audio ({len(pcm_data)} bytes)")
                self.speaker_stream.write(pcm_data)
                print("✅ Audio playback complete")
                
                # If we're hanging up and this was the final audio chunk, terminate the client
                if self.hanging_up:
                    print("📞 Final hangup audio played, terminating client...")
                    self.running = False
            else:
                print("Speaker stream unavailable - cannot play audio")
            
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
        # Make sure we clean up
        if client.websocket:
            await client.disconnect()
        client.cleanup()
    
if __name__ == "__main__":
    asyncio.run(main()) 