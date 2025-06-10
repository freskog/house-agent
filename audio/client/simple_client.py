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

# Add logging infrastructure
from utils.logging_config import setup_logging
from utils.metrics import timing_decorator, TimerContext, start_timer, end_timer
from utils.config import get_config

# Module-level logger
logger = setup_logging(__name__)

class SimpleAudioClient:
    @timing_decorator()
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
        self.tool_active = False  # Track if a tool is currently active
        self.current_tool = None  # Track the currently active tool
        
        # Setup logger for this instance
        self.logger = setup_logging(f"{__name__}.{self.__class__.__name__}")
        
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
            self.logger.warning("scipy.signal not available, audio resampling will be limited")
            self.has_signal = False
            
        self.logger.info(f"🎧 Audio client initialized for server: {server_url}")
        
    async def connect(self):
        """Connect to server and initialize audio devices"""
        with TimerContext("client_connection") as timer:
            try:
                # Connect to WebSocket server
                self.logger.info(f"🔗 Connecting to {self.server_url}...")
                timer.checkpoint("websocket_start")
                
                self.websocket = await websockets.connect(self.server_url)
                timer.checkpoint("websocket_connected")
                self.logger.info("🔗 Connected to server")
                
                # Initialize PyAudio
                self.pyaudio = pyaudio.PyAudio()
                timer.checkpoint("pyaudio_initialized")
                
                # Set up microphone input stream
                self.logger.debug("🎤 Initializing microphone...")
                self.mic_stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.mic_sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size
                )
                timer.checkpoint("microphone_initialized")
                
                # Set up speaker output stream
                self.logger.debug("🔊 Initializing speaker...")
                self.speaker_stream = self.pyaudio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.speaker_sample_rate,
                    output=True,
                    frames_per_buffer=self.chunk_size
                )
                timer.checkpoint("speaker_initialized")
                
                connection_time = timer.end()
                self.logger.info(f"🎧 Audio devices initialized in {connection_time:.2f}ms")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Connection failed: {e}")
                if self.websocket:
                    await self.websocket.close()
                self.cleanup()
                return False
    
    def cleanup(self):
        """Clean up resources"""
        self.logger.debug("🧹 Cleaning up audio resources...")
            
        if self.mic_stream:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except Exception as e:
                self.logger.warning(f"Error closing mic stream: {e}")
            finally:
                self.mic_stream = None
            
        if self.speaker_stream:
            try:
                self.speaker_stream.stop_stream()
                self.speaker_stream.close()
            except Exception as e:
                self.logger.warning(f"Error closing speaker stream: {e}")
            finally:
                self.speaker_stream = None
            
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception as e:
                self.logger.warning(f"Error terminating PyAudio: {e}")
            finally:
                self.pyaudio = None
        
        self.logger.info("🧹 Audio resources cleaned up")
    
    async def disconnect(self):
        """Disconnect from server and clean up"""
        self.logger.info("🔌 Disconnecting from server...")
        
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
                    self.logger.debug("Sent disconnect message")
                except Exception as e:
                    self.logger.warning(f"Error sending disconnect message: {e}")
                
                # Close the websocket
                try:
                    await self.websocket.close()
                except Exception as e:
                    self.logger.warning(f"Error closing websocket: {e}")
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
        finally:
            self.websocket = None
            
        self.logger.info("🔌 Disconnected from server")
            
    async def run(self):
        """Run the client"""
        if not await self.connect():
            return
            
        self.running = True
        
        self.logger.info("\n========== EDGE DEVICE SIMULATOR ==========")
        self.logger.info("Streaming microphone audio to server...")
        self.logger.info("Receiving audio will play through speakers")
        self.logger.info("\n⚠️  IMPORTANT: Use headphones to prevent echo!")
        self.logger.info("==========================================")
        self.logger.info("Press Ctrl+C to exit\n")
        
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
                self.logger.warning("Some tasks did not terminate gracefully")
                
        except asyncio.CancelledError:
            self.logger.warning("Tasks cancelled")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.logger.debug("Client shutting down...")
            self.running = False
            await self.disconnect()
            self.cleanup()
            self.logger.info("Client shutdown complete")
            
    async def send_audio_loop(self):
        """Continuously read from microphone and send to server"""
        try:
            while self.running and self.websocket:
                # If we're hanging up, stop sending audio
                if self.hanging_up:
                    self.logger.info("Hanging up - stopped sending audio")
                    self.running = False  # Set running to false to trigger client shutdown
                    break
                    
                # Check if the websocket is closed
                if hasattr(self.websocket, 'closed') and self.websocket.closed:
                    self.logger.info("WebSocket connection closed - stopped sending audio")
                    break
                    
                # Read audio data from microphone
                try:
                    audio_data = self.mic_stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Simple audio level check
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    audio_level = np.abs(audio_np).mean()
                    
                    # Only warn if audio level is very low and not too frequently
                    if audio_level < 10 and self.sequence % 100 == 0:
                        self.logger.warning(f"🎤 Very low audio level ({audio_level:.1f}). Check your microphone.")
                    
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
                        self.logger.info("Connection closed while sending audio - stopping")
                        self.running = False
                        break
                    
                    # Pace ourselves according to the chunk duration
                    chunk_duration = self.chunk_size / self.mic_sample_rate
                    await asyncio.sleep(chunk_duration * 0.5)  # Sleep for half the chunk duration
                    
                except Exception as e:
                    self.logger.error(f"Error capturing audio: {e}")
                    await asyncio.sleep(0.1)
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.debug("Connection closed - audio sending stopped")
            self.running = False
        except Exception as e:
            self.logger.warning(f"Error in send loop: {e}")
            self.running = False
            
    async def receive_audio_loop(self):
        """Receive audio and other messages from server"""
        try:
            while self.running and self.websocket:
                # Check if the websocket is closed
                if hasattr(self.websocket, 'closed') and self.websocket.closed:
                    self.logger.debug("WebSocket connection closed - stopped receiving")
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
                    self.logger.debug("Connection closed while waiting for messages")
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
                            self.logger.debug("📞 Received audio with hangup flag")
                            self.hanging_up = True
                            
                            # If this is a final empty message with hangup flag, terminate immediately
                            if is_final and "audio" not in msg["payload"]:
                                self.logger.info("📞 Received final hangup signal without audio, terminating client...")
                                self.running = False
                                break
                        
                        # Check if there's audio data to play
                        if "audio" in msg["payload"]:
                            # Decode audio data from base64
                            audio_data = base64.b64decode(msg["payload"]["audio"])
                            
                            # Play it directly - no queuing
                            if audio_data and len(audio_data) > 0:
                                self.logger.debug(f"Playing audio: hangup={is_hangup}, final={is_final}, size={len(audio_data)}bytes")
                                self.process_and_play_audio(audio_data)
                                
                                # Check if this was the final audio chunk in a hangup sequence
                                if is_hangup and is_final:
                                    self.logger.info("📞 Received final hangup audio, client will terminate after playback")
                            else:
                                self.logger.debug("Received empty audio data")
                    
                    # Handle status messages (transcription, state updates)
                    elif msg["type"] == "status":
                        # Check if we're hanging up
                        if msg["payload"].get("state") == "hanging_up":
                            self.logger.info("Server is hanging up")
                            self.hanging_up = True
                            
                        # Handle speech detection
                        if "is_speech" in msg["payload"]:
                            is_speech = msg["payload"]["is_speech"]
                            if is_speech:
                                self.logger.info("🎤 Speech detected...")
                                
                        # Handle state transitions
                        if "state" in msg["payload"]:
                            state = msg["payload"]["state"]
                            self.logger.debug(f"🔄 State: {state}")
                            
                        # Handle transcription
                        if "transcription" in msg["payload"]:
                            text = msg["payload"]["transcription"]["text"]
                            self.logger.info(f"🔊 You said: \"{text}\"")
                    
                    # Handle error messages
                    elif msg["type"] == "error":
                        error = msg["payload"].get("error", "Unknown error")
                        self.logger.error(f"❌ Error: {error}")
                        
                    # Handle tool events
                    elif msg["type"] == "tool_event":
                        event_type = msg["payload"].get("event_type")
                        tool_name = msg["payload"].get("tool_name")
                        details = msg["payload"].get("details")
                        
                        if event_type == "tool_start":
                            self.tool_active = True
                            self.current_tool = tool_name
                            self.logger.info(f"🔧 Tool started: {tool_name}")
                            # TODO: Start blinking LEDs here
                            
                        elif event_type == "tool_end":
                            if tool_name == self.current_tool:
                                self.tool_active = False
                                self.current_tool = None
                                self.logger.info(f"✅ Tool completed: {tool_name}")
                                # TODO: Stop blinking LEDs here if no other tools are active
                                
                        elif event_type == "tool_error":
                            error_msg = details.get("error") if details else "Unknown error"
                            self.logger.error(f"❌ Tool error in {tool_name}: {error_msg}")
                            if tool_name == self.current_tool:
                                self.tool_active = False
                                self.current_tool = None
                                # TODO: Show error pattern on LEDs
                                
                        # Reset LED state when transcription starts
                        if msg["type"] == "status" and msg["payload"].get("state") == "transcribing":
                            self.tool_active = False
                            self.current_tool = None
                            # TODO: Reset LEDs to idle state
                
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid JSON message")
                except KeyError as e:
                    self.logger.error(f"Missing key in message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    import traceback
                    traceback.print_exc()
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.debug("Connection closed - message receiving stopped")
            self.running = False
        except Exception as e:
            self.logger.error(f"Error in receive loop: {e}")
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
                self.logger.debug(f"▶️ Playing audio ({len(pcm_data)} bytes)")
                self.speaker_stream.write(pcm_data)
                self.logger.debug("✅ Audio playback complete")
                
                # If we're hanging up and this was the final audio chunk, terminate the client
                if self.hanging_up:
                    self.logger.info("📞 Final hangup audio played, terminating client...")
                    self.running = False
            else:
                self.logger.warning("Speaker stream unavailable - cannot play audio")
            
        except Exception as audio_err:
            self.logger.error(f"Error processing audio: {audio_err}")
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
        client.logger.info("\n🛑 Interrupted by user")
    finally:
        # Make sure we clean up
        if client.websocket:
            await client.disconnect()
        client.cleanup()
    
if __name__ == "__main__":
    asyncio.run(main()) 