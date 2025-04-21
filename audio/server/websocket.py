"""
WebSocket server for handling audio connections and streaming.
"""

import asyncio
import websockets
import json
import time
import base64
import numpy as np
import random
import collections
import wave
import os
import uuid
from typing import Optional, Dict, Any, Callable, Set, List, Awaitable, Union
from .protocol import (
    Message, MessageType, AudioConfig, DEFAULT_AUDIO_CONFIG,
    AudioStreamPayload, AudioPlaybackPayload, StatusPayload, ErrorPayload
)
from .vad import VADHandler, VADConfig, VADResult
from .transcribe import Transcriber, TranscriptionConfig, TranscriptionResult
from .tts import TTSEngine, TTSConfig
from langsmith import traceable

# Create a LangSmith client for additional tracing needs
# Importing Client for tracing
try:
    from langsmith import Client
    # Don't initialize at module level - we'll create clients on demand
    has_langsmith = True
    print(f"LangSmith module imported successfully")
except ImportError:
    # Fallback if langsmith is not available
    print("LangSmith not available, tracing disabled")
    has_langsmith = False

def create_langsmith_client():
    """Create a LangSmith client with current environment variables"""
    if not has_langsmith:
        return None
        
    try:
        from langsmith import Client
        # Force the client to accept API keys with any prefix
        api_key = os.environ.get("LANGSMITH_API_KEY")
        if api_key:
            # The client seems to check if the key starts with 'ls-' but our key starts with 'lsv2_'
            # Let's temporarily patch the environment to disable validation
            os.environ["LANGSMITH_ALLOW_ANY_API_KEY_FORMAT"] = "true"
            
        client = Client()
        has_trace = hasattr(client, 'trace')
        print(f"LangSmith client created. has_trace={has_trace}, API Key set: {bool(os.environ.get('LANGSMITH_API_KEY'))}, Tracing enabled: {os.environ.get('LANGSMITH_TRACING')}")
        return client
    except Exception as e:
        print(f"Error creating LangSmith client: {e}")
        return None

# Type for transcription callback
# Takes transcription result, returns text response or audio bytes (or both in the future)
TranscriptionCallback = Callable[[TranscriptionResult], Awaitable[Union[str, bytes, Dict[str, Any]]]]

class AudioProcessingState:
    """Tracks the state of audio processing for a client"""
    
    def __init__(self):
        self.is_speech_active = False
        self.speech_frames: List[bytes] = []
        self.last_speech_time = time.time()  # Initialize with current time to avoid NoneType error
        self.recording_start_time = time.time()  # Initialize with current time
        self.pre_vad_buffer = collections.deque(maxlen=10)  # Increase buffer size to ~0.6 seconds at 16kHz
        # Generate a unique thread ID for this client to use with LangSmith
        self.thread_id = str(uuid.uuid4())
        print(f"New client state initialized with thread_id: {self.thread_id}")
        
    def reset(self):
        """Reset the state"""
        self.is_speech_active = False
        # Keep speech frames for debugging - they'll be replaced on next speech detection
        self.last_speech_time = time.time()
        self.recording_start_time = time.time()
        # Don't clear pre_vad_buffer - we want to keep recent audio
        # Don't reset thread_id - keep the same thread for the entire client session
        
    def add_speech_frame(self, frame: bytes):
        """Add a frame to the speech buffer"""
        if not self.is_speech_active:
            self.is_speech_active = True
            self.recording_start_time = time.time()
            
        if frame and len(frame) > 0:
            self.speech_frames.append(frame)
            self.last_speech_time = time.time()
        else:
            print("WARNING: Attempted to add empty frame to speech buffer")
        
    def get_audio_data(self) -> bytes:
        """Get the combined audio data"""
        if not self.speech_frames:
            print("WARNING: No speech frames to join!")
            return b''
            
        # Filter out empty frames
        non_empty_frames = [frame for frame in self.speech_frames if frame and len(frame) > 0]
        if len(non_empty_frames) < len(self.speech_frames):
            print(f"WARNING: Filtered out {len(self.speech_frames) - len(non_empty_frames)} empty frames")
            
        if not non_empty_frames:
            print("ERROR: All frames were empty!")
            return b''
            
        return b''.join(non_empty_frames)
        
    def get_recording_duration(self) -> float:
        """Get the recording duration"""
        if self.recording_start_time is None:
            return 0.0
        return time.time() - self.recording_start_time

class AudioServer:
    """WebSocket server that handles audio connections and processing"""
    
    def __init__(self, 
                host: str = "localhost", 
                port: int = 8765,
                vad_config: Optional[VADConfig] = None,
                transcription_config: Optional[TranscriptionConfig] = None,
                tts_config: Optional[TTSConfig] = None,
                transcription_callback: Optional[TranscriptionCallback] = None,
                save_recordings: bool = False):
        self.host = host
        self.port = port
        
        # Create default VAD config if none was provided
        if vad_config is None:
            # Use much more aggressive VAD config to significantly reduce false positives
            vad_config = VADConfig(
                threshold=0.5,                 # Much higher threshold to dramatically reduce sensitivity
                min_speech_duration=0.8,       # Require longer speech segments to consider it valid
                min_silence_duration=1.0,      # Require more definitive silence between utterances
                speech_pad_ms=80,              # Reduced padding to minimize detecting non-speech
                buffer_size=8,                 # Smaller buffer to accumulate less potential noise
                verbose=False                  # Keep logging minimal
            )
        
        self.vad_handler = VADHandler(config=vad_config)
        self.transcriber = Transcriber(config=transcription_config)
        self.tts_engine = TTSEngine(config=tts_config)
        self.connections: Set[websockets.WebSocketServerProtocol] = set()
        self.client_states: Dict[websockets.WebSocketServerProtocol, AudioProcessingState] = {}
        self.sequence = 0
        self.silence_threshold = 1.5  # Increased from 1.0 to 1.5 seconds of silence before processing
        self.max_recording_duration = 30.0  # maximum recording duration
        self.save_recordings = save_recordings  # Flag to control recording audio to disk
        
        # Store the callback
        self.transcription_callback = transcription_callback or self._default_transcription_callback
        
        # All components should already be initialized eagerly at this point,
        # but we'll verify and report their status here
        print("Checking audio processing components initialization...")
        
        vad_initialized = self.vad_handler.initialized
        transcriber_initialized = self.transcriber.model is not None
        tts_initialized = self.tts_engine.initialized
        
        if not vad_initialized:
            print("⚠️ WARNING: VAD engine not initialized")
        if not transcriber_initialized:
            print("⚠️ WARNING: Transcription engine not initialized")
        if not tts_initialized:
            print("⚠️ WARNING: TTS engine not initialized")
            
        if vad_initialized and transcriber_initialized and tts_initialized:
            print("✅ All audio processing components initialized successfully")
        else:
            print("⚠️ Some components failed to initialize, functionality may be limited")
        
    async def start(self):
        """Start the WebSocket server"""
        # Start server with specific ping/pong settings
        self.server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            ping_interval=20,    # Send a ping every 20 seconds
            ping_timeout=10,     # Wait 10 seconds for a pong response
            close_timeout=5      # Wait 5 seconds for the close handshake
        )
        print(f"Audio server started on ws://{self.host}:{self.port}")
        return self.server
        
    async def stop(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
    async def handle_connection(self, websocket):
        """Handle a new WebSocket connection"""
        try:
            # Add connection to set
            self.connections.add(websocket)
            # Create state for this client
            self.client_states[websocket] = AudioProcessingState()
            
            print(f"New connection from {websocket.remote_address}")
            
            async for message in websocket:
                try:
                    # Parse incoming message
                    msg = Message.from_json(message)
                    
                    # Handle different message types
                    if msg.type == MessageType.AUDIO_STREAM:
                        await self.handle_audio_stream(websocket, msg)
                    elif msg.type == MessageType.DISCONNECT:
                        # Client is explicitly disconnecting
                        print(f"Client {websocket.remote_address} sent disconnect message")
                        break
                    else:
                        print(f"Unhandled message type: {msg.type}")
                        
                except json.JSONDecodeError:
                    print("Invalid message format")
                except Exception as e:
                    print(f"Error handling message: {e}")
                    await self.send_error(websocket, str(e))
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"Connection closed from {websocket.remote_address}: {e}")
        finally:
            # Cleanup when client disconnects for any reason
            print(f"Cleaning up resources for {websocket.remote_address}")
            if websocket in self.connections:
                self.connections.remove(websocket)
            if websocket in self.client_states:
                del self.client_states[websocket]
            
    async def handle_audio_stream(self, websocket, message: Message):
        """Handle incoming audio stream data"""
        # First check if connection is still valid before processing
        if websocket not in self.connections:
            print("Ignoring audio stream from closed connection")
            return
            
        try:
            # Extract audio data from payload
            audio_data_b64 = message.payload.get("audio")
            if not audio_data_b64:
                await self.send_error(websocket, "Missing audio data")
                return
                
            # Decode audio data from base64
            try:
                audio_data = base64.b64decode(audio_data_b64)
                if len(audio_data) == 0:
                    print("WARNING: Received empty audio data from client!")
                    return  # Skip empty frames entirely
            except Exception as e:
                print(f"ERROR: Failed to decode base64 audio data: {e}")
                await self.send_error(websocket, f"Invalid audio data format: {e}")
                return
            
            # Check again if connection is still valid after decoding
            if websocket not in self.connections:
                print("Connection closed during audio processing, skipping further processing")
                return
                
            # Get client state
            client_state = self.client_states.get(websocket)
            if not client_state:
                print("Client state not found, ignoring audio frame")
                return
            
            # Occasionally analyze the raw chunk to check for sample rate issues
            if random.random() < 0.05:  # Only analyze ~5% of chunks to reduce overhead
                try:
                    # Convert to numpy for analysis
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                except Exception as e:
                    print(f"Error analyzing raw audio chunk: {e}")
                
            # Add to pre-VAD buffer
            if len(audio_data) > 0:
                # Only store non-empty frames in pre-buffer
                client_state.pre_vad_buffer.append(audio_data)
            
            # Process through VAD
            vad_result = self.vad_handler.process_chunk(audio_data)
            
            # Check for state transitions and send status updates only on changes
            was_speech_active = client_state.is_speech_active
            
            # Handle speech detection
            current_time = time.time()
            
            if vad_result.is_speech:
                # Initialize speech frames collection only on transition from silence to speech
                if not was_speech_active:
                    # Start fresh with the pre-VAD buffer (these are frames from just before speech started)
                    # Copy frames from pre-VAD buffer to ensure we don't lose the beginning of speech
                    client_state.is_speech_active = True
                    client_state.speech_frames = []  # Start with a fresh collection
                    client_state.recording_start_time = time.time()
                    
                    # Add all non-empty frames from pre-VAD buffer (silently skip empty ones)
                    for frame in client_state.pre_vad_buffer:
                        if frame and len(frame) > 0:
                            client_state.speech_frames.append(frame)
                    
                    # Only log a simple message when speech starts
                    print("Speech detected - recording started")
                
                # Always add the current frame to speech collection
                if audio_data and len(audio_data) > 0:
                    client_state.speech_frames.append(audio_data)
                    client_state.last_speech_time = current_time
                    
                    # Reduce logging frequency even further (only log every 100 frames)
                    if len(client_state.speech_frames) % 100 == 0:
                        print(f"Recording: {len(client_state.speech_frames)*2/16000:.1f}s")
                
                # If this is a transition from silence to speech, send status
                if not was_speech_active:
                    await self.send_status(websocket, vad_result)
                
                # Check if we've exceeded max recording duration
                if client_state.get_recording_duration() > self.max_recording_duration:
                    print(f"Max recording duration reached ({self.max_recording_duration}s), processing...")
                    await self.send_status(websocket, vad_result, state="processing")
                    await self.process_recording(websocket, client_state)
            elif client_state.is_speech_active:
                # We're in speech mode but VAD reports silence
                # Still collect frames to avoid gaps during brief silences
                client_state.speech_frames.append(audio_data)
                
                # Check if silence duration exceeds threshold
                silence_duration = current_time - client_state.last_speech_time
                
                # Only log when silence exceeds threshold
                if silence_duration > self.silence_threshold:
                    print("Processing recording...")
                    await self.send_status(websocket, vad_result, state="processing")
                    await self.process_recording(websocket, client_state)
                    if hasattr(client_state, '_silence_logged'):
                        delattr(client_state, '_silence_logged')
            else:
                # If this is a transition from speech to silence (after processing),
                # send a status update
                if was_speech_active:
                    print(f"Speech ended")
                    await self.send_status(websocket, vad_result)
                    if hasattr(client_state, '_silence_logged'):
                        delattr(client_state, '_silence_logged')
        
        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed during audio processing")
            # Clean up resources for this connection
            if websocket in self.connections:
                self.connections.remove(websocket)
            if websocket in self.client_states:
                del self.client_states[websocket]
        except Exception as e:
            print(f"Error processing audio stream: {e}")
            try:
                if websocket in self.connections:
                    await self.send_error(websocket, str(e))
            except websockets.exceptions.ConnectionClosed:
                # If the connection is already closed, remove it from our tracking
                if websocket in self.connections:
                    self.connections.remove(websocket)
                if websocket in self.client_states:
                    del self.client_states[websocket]
            except Exception as nested_error:
                print(f"Error sending error about audio processing: {nested_error}")
            
    @traceable(run_type="chain", name="Voice_Assistant_Pipeline")
    async def process_recording(self, websocket, client_state: AudioProcessingState):
        """Process a completed recording

        This function handles the complete voice pipeline:
        1. Gets the combined audio data
        2. Transcribes it using Whisper
        3. Processes the transcription with the callback
        4. Converts any text response to speech
        5. Sends the response back to the client
        """
        try:
            # Get the combined audio data
            audio_data = client_state.get_audio_data()
            recording_duration = client_state.get_recording_duration()
            
            # Check if we have enough data
            if not audio_data or len(audio_data) < 1000:  # Less than 1KB is probably not speech
                # Only log for recordings that are not tiny noise blips
                if len(audio_data) > 500:
                    print(f"WARNING: Recording too short to process ({len(audio_data)} bytes)")
                # Reset this recording
                client_state.reset()
                # Clear speech indicator
                await self.send_status(websocket, VADResult(is_speech=False), "idle")
                return
                
            # More concise logging
            print(f"Processing: {recording_duration:.1f}s")
            
            # Optional: save recording to disk for debugging
            if self.save_recordings:
                try:
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Create recordings directory if it doesn't exist
                    os.makedirs("recordings", exist_ok=True)
                    
                    # Write to WAV file
                    with wave.open(f"recordings/recording_{timestamp}.wav", "wb") as wf:
                        wf.setnchannels(1)  # Mono
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(16000)  # 16kHz
                        wf.writeframes(audio_data)
                        
                    print(f"Saved recording to recordings/recording_{timestamp}.wav")
                except Exception as e:
                    print(f"Error saving recording: {e}")
            
            # Add trace metadata for LangSmith
            audio_metadata = {
                "audio_size_kb": len(audio_data) / 1024,
                "recording_duration_sec": recording_duration,
                "timestamp": time.time(),
                "thread_id": client_state.thread_id,  # Include thread ID in metadata
                "client_id": str(id(websocket))  # Add unique client identifier
            }
            
            # Create a fresh LangSmith client to ensure we have the latest environment variables
            os.environ["LANGSMITH_ALLOW_ANY_API_KEY_FORMAT"] = "true"  # Allow lsv2_ format
            langsmith_client = create_langsmith_client()
            
            # Define a trace context
            if langsmith_client and hasattr(langsmith_client, 'trace'):
                try:
                    print(f"Creating LangSmith trace with thread_id: {client_state.thread_id}")
                    # Pass thread_id to ensure traces from the same client are grouped together
                    trace_context = langsmith_client.trace(
                        "Voice_Processing", 
                        metadata=audio_metadata,
                        thread_id=client_state.thread_id
                    )
                except Exception as e:
                    print(f"Error creating LangSmith trace: {e}")
                    from contextlib import nullcontext
                    trace_context = nullcontext()
            else:
                # Use a dummy context manager when tracing is not available
                from contextlib import nullcontext
                trace_context = nullcontext()
                if not langsmith_client:
                    print("LangSmith client not available")
                elif not hasattr(langsmith_client, 'trace'):
                    print("LangSmith client does not have trace method")
                
            # Trace this part of the process to LangSmith if available
            with trace_context as parent_run:
                # Send processing status to client
                await self.send_status(websocket, VADResult(is_speech=False), "processing")
                await self.send_processing_indicator(websocket)
                
                # Perform transcription
                transcription_start_time = time.time()
                transcription_result = self.transcriber.transcribe_audio(
                    audio_data, 
                    thread_id=client_state.thread_id
                )
                transcription_time = time.time() - transcription_start_time
                print(f"Transcribed ({transcription_time:.1f}s): '{transcription_result.text}'")
                
                # Reset recording state
                client_state.reset()
                
                # Send transcription to client
                try:
                    await self.send_transcription(websocket, transcription_result)
                except Exception as e:
                    print(f"Error sending transcription: {e}")
                
                # Add trace metadata for transcription results
                if parent_run and hasattr(parent_run, 'add_metadata'):
                    parent_run.add_metadata({
                        "transcription_text": transcription_result.text,
                        "transcription_duration_sec": transcription_time
                    })
                
                # Process transcription with callback
                processing_start_time = time.time()
                
                # Use the configured callback or default
                callback_result = await self.transcription_callback(transcription_result)
                processing_time = time.time() - processing_start_time
                
                # Debug the callback result
                print(f"DEBUG: Callback result type: {type(callback_result)}, value: {callback_result}")
                
                # Handle different types of callback results
                if isinstance(callback_result, str):
                    # Text response
                    response_text = callback_result
                    print(f"DEBUG: Got text response: '{response_text}', type: {type(response_text)}, length: {len(str(response_text))}")
                    
                    # Add trace metadata for the response if available
                    if parent_run and hasattr(parent_run, 'add_metadata'):
                        parent_run.add_metadata({
                            "response_text": response_text
                        })
                    
                    # Check for valid text more carefully
                    is_valid_text = bool(response_text and response_text.strip())
                    print(f"DEBUG: Is valid text for TTS? {is_valid_text}")
                        
                    # Check if the response is a question
                    is_question = self._is_question(response_text) if is_valid_text else False
                    print(f"DEBUG: Is response a question? {is_question}")
                        
                    # Process TTS regardless of tracing availability
                    if is_valid_text:
                        print(f"DEBUG: Starting TTS for: '{response_text}'")
                        tts_start_time = time.time()
                        
                        # Use streaming TTS to send audio as soon as each sentence is ready
                        streaming_success = await self.text_to_speech_streaming(response_text, websocket)
                        tts_time = time.time() - tts_start_time
                        
                        if streaming_success:
                            print(f"DEBUG: TTS streaming success in {tts_time:.1f}s")
                            
                            # Add trace metadata for TTS results if available
                            if parent_run and hasattr(parent_run, 'add_metadata'):
                                parent_run.add_metadata({
                                    "tts_duration_sec": tts_time,
                                    "tts_streaming": True
                                })
                            
                            # If the response is not a question, hang up the call
                            if not is_question:
                                print(f"DEBUG: Response is not a question. Hanging up automatically...")
                                # Small delay to ensure audio is played before hanging up
                                await asyncio.sleep(0.5)
                                await self.hang_up(websocket, "")  # Empty message to avoid saying "Call ended" after response
                        else:
                            print(f"DEBUG: TTS streaming failed - falling back to non-streaming TTS")
                            # Fallback to non-streaming TTS
                            audio_response = await self.text_to_speech(response_text)
                            
                            if audio_response:
                                print(f"DEBUG: Fallback TTS success - audio size: {len(audio_response)/1024:.1f}KB")
                                
                                # Add trace metadata for TTS results if available
                                if parent_run and hasattr(parent_run, 'add_metadata'):
                                    parent_run.add_metadata({
                                        "tts_duration_sec": tts_time,
                                        "tts_audio_size_kb": len(audio_response) / 1024,
                                        "tts_streaming": False,
                                        "tts_fallback": True
                                    })
                                
                                # Send audio back to client
                                print(f"DEBUG: Sending fallback audio to client")
                                await self.send_audio_playback(websocket, audio_response)
                                print(f"DEBUG: Fallback audio sent successfully")
                                
                                # If the response is not a question, hang up the call
                                if not is_question:
                                    print(f"DEBUG: Response is not a question. Hanging up automatically...")
                                    # Small delay to ensure audio is played before hanging up
                                    await asyncio.sleep(0.5)
                                    await self.hang_up(websocket, "")  # Empty message to avoid saying "Call ended" after response
                            else:
                                print(f"DEBUG: Fallback TTS also failed - returned empty audio for: '{response_text}'")
                                await self.send_error(websocket, "Failed to generate speech")
                    else:
                        print(f"DEBUG: Empty or invalid response text: '{response_text}', not generating TTS")
                        
                elif isinstance(callback_result, bytes):
                    # Direct audio response - log minimally
                    print("Got direct audio response")
                    await self.send_audio_playback(websocket, callback_result)
                    
                elif isinstance(callback_result, dict):
                    # Structured response - log minimally
                    print("Got structured response")
                    
                    # Handle text for TTS
                    if "text" in callback_result and callback_result["text"]:
                        text_response = callback_result["text"]
                        print(f"DEBUG: Got structured response text: '{text_response}'")
                        
                        # Check if the response is a question
                        is_question = self._is_question(text_response)
                        print(f"DEBUG: Is structured response a question? {is_question}")
                        
                        # Add trace metadata for the response if available
                        if parent_run and hasattr(parent_run, 'add_metadata'):
                            parent_run.add_metadata({
                                "response_text": text_response
                            })
                        
                        # Convert to speech (regardless of tracing)
                        tts_start_time = time.time()
                        
                        # Use streaming TTS for structured responses too
                        streaming_success = await self.text_to_speech_streaming(text_response, websocket)
                        tts_time = time.time() - tts_start_time
                        
                        if streaming_success:
                            print(f"DEBUG: TTS streaming success for structured response in {tts_time:.1f}s")
                            
                            # Add trace metadata for TTS results if available
                            if parent_run and hasattr(parent_run, 'add_metadata'):
                                parent_run.add_metadata({
                                    "tts_duration_sec": tts_time,
                                    "tts_streaming": True
                                })
                            
                            # If the response is not a question, hang up the call
                            if not is_question:
                                print(f"DEBUG: Structured response is not a question. Hanging up automatically...")
                                # Small delay to ensure audio is played before hanging up
                                await asyncio.sleep(0.5)
                                await self.hang_up(websocket, "")  # Empty message to avoid saying "Call ended" after response
                        else:
                            print(f"DEBUG: TTS streaming failed for structured response - falling back to non-streaming TTS")
                            # Fallback to non-streaming TTS
                            audio_response = await self.text_to_speech(text_response)
                            
                            if audio_response:
                                print(f"DEBUG: Fallback TTS success - structured response audio size: {len(audio_response)/1024:.1f}KB")
                                
                                # Add trace metadata for TTS results if available
                                if parent_run and hasattr(parent_run, 'add_metadata'):
                                    parent_run.add_metadata({
                                        "tts_duration_sec": tts_time,
                                        "tts_audio_size_kb": len(audio_response) / 1024,
                                        "tts_streaming": False,
                                        "tts_fallback": True
                                    })
                                
                                # Send audio response
                                await self.send_audio_playback(websocket, audio_response)
                                
                                # If the response is not a question, hang up the call
                                if not is_question:
                                    print(f"DEBUG: Structured response is not a question. Hanging up automatically...")
                                    # Small delay to ensure audio is played before hanging up
                                    await asyncio.sleep(0.5)
                                    await self.hang_up(websocket, "")  # Empty message to avoid saying "Call ended" after response
                            else:
                                print(f"DEBUG: TTS failed for structured response text: '{text_response}'")
                                await self.send_error(websocket, "Failed to generate speech")

                    # Handle direct audio if provided
                    elif "audio" in callback_result and callback_result["audio"]:
                        await self.send_audio_playback(websocket, callback_result["audio"])
                        
                    # Handle status updates
                    if "status" in callback_result:
                        await self.send_status(websocket, VADResult(), callback_result["status"])
                        
                else:
                    print(f"Unknown callback result type: {type(callback_result)}")
                
                # Finally, send status back to idle when done processing
                await self.send_status(websocket, VADResult(is_speech=False), "idle")
                
                # Record total processing duration
                total_duration = time.time() - transcription_start_time
                if parent_run and hasattr(parent_run, 'add_metadata'):
                    parent_run.add_metadata({
                        "total_processing_duration_sec": total_duration
                    })
                
        except Exception as e:
            print(f"Error processing recording: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to send error to client
            try:
                await self.send_error(websocket, f"Processing error: {str(e)}")
                await self.send_status(websocket, VADResult(is_speech=False), "idle")
            except Exception:
                pass
                
            # Reset state
            client_state.reset()

    async def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech using TTS engine"""
        try:
            if not text or text.strip() == "":
                print("Warning: Empty text provided to TTS, skipping synthesis")
                return b''
                
            print(f"TTS request: '{text}'")
            
            # Check if TTS engine is initialized
            if not self.tts_engine or not self.tts_engine.is_initialized():
                print("ERROR: TTS engine is not initialized!")
                return b''
                
            # Use the TTS engine to synthesize speech
            audio_data = await self.tts_engine.synthesize(text)
            
            if audio_data is None:
                print("ERROR: TTS returned None!")
                return b''
                
            print(f"TTS response size: {len(audio_data)} bytes")
            if len(audio_data) < 100:
                print("Warning: TTS produced very small or empty audio data")
            
            return audio_data
        except Exception as e:
            print(f"Error in TTS processing: {e}")
            import traceback
            traceback.print_exc()
            # Return empty audio as fallback
            return b''
            
    async def _default_transcription_callback(self, transcription: TranscriptionResult) -> str:
        """Default callback that just echoes the transcription"""
        return f"I heard: {transcription.text}"
            
    async def send_status(self, websocket, vad_result: VADResult, state: Optional[str] = None):
        """Send status update to client"""
        try:
            # Check if connection is still open
            if websocket not in self.connections:
                print(f"Not sending status update to closed connection")
                return
                
            # Prepare payload
            payload = {
                "is_speech": vad_result.is_speech,
                "confidence": float(vad_result.confidence)
            }
            
            # Add state if provided
            if state:
                payload["state"] = state
                
            # Get or initialize sequence number
            sequence = 0
            if hasattr(websocket, 'next_sequence'):
                sequence = websocket.next_sequence
                websocket.next_sequence += 1
            else:
                # Initialize sequence counter if it doesn't exist
                websocket.next_sequence = 1
                
            # Create message
            message = Message(
                type=MessageType.STATUS,
                sequence=sequence,
                payload=payload
            )
            
            # Check again if connection is still open
            if websocket not in self.connections:
                print(f"Connection closed before sending status")
                return
                
            # Send message
            await websocket.send(message.to_json())
        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed while sending status update")
            # Clean up this connection
            if websocket in self.connections:
                self.connections.remove(websocket)
            if websocket in self.client_states:
                del self.client_states[websocket]
        except Exception as e:
            print(f"Error sending status: {e}")
            # Don't try to send an error about failing to send status - would be circular
            
    async def send_transcription(self, websocket, result: TranscriptionResult):
        """Send transcription to client"""
        try:
            # Check if connection is still open
            if websocket not in self.connections:
                print(f"Not sending transcription to closed connection")
                return
                
            # Prepare payload
            payload = {
                "transcription": {
                    "text": result.text,
                    "confidence": result.confidence
                }
            }
            
            # Add language if available
            if result.language:
                payload["transcription"]["language"] = result.language
                
            # Get or initialize sequence number
            sequence = 0
            if hasattr(websocket, 'next_sequence'):
                sequence = websocket.next_sequence
                websocket.next_sequence += 1
            else:
                # Initialize sequence counter if it doesn't exist
                websocket.next_sequence = 1
                
            # Create message
            message = Message(
                type=MessageType.STATUS,
                sequence=sequence,
                payload=payload
            )
            
            # Double-check connection is still open
            if websocket not in self.connections:
                print(f"Connection closed before sending transcription")
                return
                
            # Send message
            await websocket.send(message.to_json())
        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed while sending transcription")
            # Clean up this connection
            if websocket in self.connections:
                self.connections.remove(websocket)
            if websocket in self.client_states:
                del self.client_states[websocket]
        except Exception as e:
            print(f"Error sending transcription: {e}")
            # Don't try to send an error about sending a transcription
            
    async def send_error(self, websocket, error: str, code: Optional[int] = None):
        """Send an error message to the client"""
        try:
            # Check if the websocket is still valid and in our connections set
            if websocket not in self.connections:
                print(f"Not sending error to closed connection: {error}")
                return
                
            # Prepare payload
            payload = {
                "error": error
            }
            
            # Add code if provided
            if code:
                payload["code"] = code
                
            # Get or initialize sequence number
            sequence = 0
            if hasattr(websocket, 'next_sequence'):
                sequence = websocket.next_sequence
                websocket.next_sequence += 1
            else:
                # Initialize sequence counter if it doesn't exist
                websocket.next_sequence = 1
                
            # Create message
            message = Message(
                type=MessageType.ERROR,
                sequence=sequence,
                payload=payload
            )
            
            # Check connection state again before sending
            if websocket not in self.connections:
                print(f"Connection closed before sending error message")
                return
                
            # Send message
            await websocket.send(message.to_json())
        except websockets.exceptions.ConnectionClosed:
            # Quietly handle closed connections
            if websocket in self.connections:
                self.connections.remove(websocket)
            if websocket in self.client_states:
                del self.client_states[websocket]
        except Exception as e:
            print(f"Error sending error message: {e}")
            # Don't try to send an error about sending an error - would cause a loop
            
    async def send_audio_playback(self, websocket, audio_data: bytes, is_hangup_message=False, is_final=False):
        """Send audio data to a client for playback
        
        Args:
            websocket: The websocket connection to send to
            audio_data: The audio data to send, as raw bytes
            is_hangup_message: Whether this is the final hangup message
            is_final: Whether this is the final audio chunk in a streaming sequence
        """
        try:
            # Check if this is a valid connection
            if websocket not in self.connections:
                print(f"Warning: Attempted to send audio to closed connection")
                return False
            
            # Encode audio data to base64
            audio_b64 = ""
            if audio_data:  # Only encode if there's actual data
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Create message with sequence parameter
            sequence = 0
            if hasattr(websocket, 'next_sequence'):
                sequence = websocket.next_sequence
                websocket.next_sequence += 1
            else:
                # Initialize sequence counter if it doesn't exist
                websocket.next_sequence = 1
                
            # Create message with sequence parameter
            payload = {
                "is_final": is_final,
                "is_hangup": is_hangup_message
            }
            
            # Only include audio data if it's not empty
            if audio_b64:
                payload["audio"] = audio_b64
                
            msg = Message(
                type=MessageType.AUDIO_PLAYBACK,
                sequence=sequence,
                payload=payload
            )
            
            # Send message
            await websocket.send(msg.to_json())
            
            # Log only for empty final chunks
            if is_final and not audio_data:
                print(f"Sent final stream marker (empty chunk)")
                
            return True
        except Exception as e:
            print(f"Error sending audio playback: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    async def send_processing_indicator(self, websocket):
        """Send a very short 'processing' audio indicator to the client
        
        This creates a perception of immediate responsiveness while the actual
        response is being generated. It's a simple audio tone or click.
        """
        try:
            # Check if this is a valid connection
            if websocket not in self.connections:
                print(f"Warning: Attempted to send indicator to closed connection")
                return False
                
            # Generate a very short tone or click sound (100ms)
            # We'll create a simple sine wave beep at 440Hz (A4)
            sample_rate = 16000
            duration = 0.05  # 50ms - very short
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            tone = np.sin(2 * np.pi * 440 * t) * 0.1  # Low volume
            
            # Apply fade in/out to avoid clicks
            fade_ms = 10  # 10ms fade
            fade_samples = int(fade_ms * sample_rate / 1000)
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            # Apply fades
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out
            
            # Convert to int16 samples
            audio_data = (tone * 32767).astype(np.int16).tobytes()
            
            # Send the tone as audio playback
            return await self.send_audio_playback(websocket, audio_data)
        except Exception as e:
            print(f"Error sending processing indicator: {e}")
            return False
            
    async def hang_up(self, websocket=None, message: str = "Call ended. Goodbye!"):
        """Hang up the call with a specific client
        
        Args:
            websocket: The websocket connection to close. If None, hang up with the most recent client.
            message: A goodbye message to send before hanging up
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Hang up requested for client: {websocket.remote_address if websocket else 'None'}")
            print(f"With message: '{message}'")
            
            if websocket is None:
                if not self.connections:
                    print("No active connections to hang up")
                    return False
                # Take the most recent client if none specified
                websocket = next(iter(self.connections))
                print(f"Selected client: {websocket.remote_address}")
                
            # Check if this is a valid connection
            if websocket not in self.connections:
                print(f"Cannot hang up: Connection not found in active connections")
                print(f"Available connections: {len(self.connections)}")
                
                # Try to find by ID/remote_address as a fallback
                if hasattr(websocket, 'remote_address') and websocket.remote_address:
                    addr = websocket.remote_address
                    # Try to find a matching connection
                    for conn in self.connections:
                        if hasattr(conn, 'remote_address') and conn.remote_address == addr:
                            print(f"Found matching connection by address: {addr}")
                            websocket = conn
                            break
                    
                # If still not found after the fallback attempt
                if websocket not in self.connections:
                    print(f"Cannot hang up: Connection not found even after fallback checks")
                    return False
            
            # First, send hanging_up status to prepare client
            try:
                print("Step 1: Sending hanging_up status")
                await self.send_status(
                    websocket, 
                    VADResult(is_speech=False, confidence=0.0), 
                    state="hanging_up"
                )
                print("Hanging_up status sent")
            except websockets.exceptions.ConnectionClosed:
                print("Connection already closed, skipping hanging_up status")
                # If connection is already closed, skip to removal
                if websocket in self.connections:
                    self.connections.remove(websocket)
                if websocket in self.client_states:
                    del self.client_states[websocket]
                return True  # Already closed, so consider it a success
            except Exception as e:
                print(f"Error sending hanging_up status: {e}")
                import traceback
                traceback.print_exc()
                
            # Send a goodbye message only if not empty and connection is still open
            if message and message.strip() and websocket in self.connections:
                try:
                    print(f"Step 2: Synthesizing goodbye message: '{message}'")
                    
                    # Use streaming TTS for the goodbye message for better responsiveness
                    streaming_success = await self.text_to_speech_streaming(
                        message,
                        websocket,
                        is_hangup_message=True
                    )
                    
                    if streaming_success:
                        print("Goodbye message streamed successfully")
                    else:
                        # Check if connection is still open before fallback
                        if websocket in self.connections:
                            # Fallback to non-streaming TTS if streaming fails
                            print("Streaming TTS failed for goodbye message, falling back to regular TTS")
                            audio_data = await self.tts_engine.synthesize(message)
                            
                            # Check audio was generated
                            if not audio_data or len(audio_data) == 0:
                                print("Warning: Failed to generate audio for goodbye message")
                            else:
                                print(f"Step 3: Sending {len(audio_data)} bytes of goodbye audio")
                                # Send the audio with hangup flag for optimized delivery
                                await self.send_audio_playback(websocket, audio_data, is_hangup_message=True)
                                print("Goodbye audio sent successfully")
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed during goodbye message, continuing with cleanup")
                except Exception as e:
                    print(f"Error sending goodbye message: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Remove the connection from server-side tracking ONLY
            # But do NOT close the connection - let the client decide when to disconnect
            # after it has received and played all audio
            print(f"Step 4: Removing client from server tracking collections")
            if websocket in self.client_states:
                del self.client_states[websocket]
                print("Client state removed from tracking")
            
            # Important: We keep the connection in self.connections until the client
            # explicitly disconnects, so we can still send messages if needed
            print(f"Step 5: Hangup process complete. Client will disconnect when ready.")
            return True
            
        except Exception as e:
            print(f"Error hanging up call: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    async def broadcast(self, message: Message):
        """Broadcast a message to all connected clients"""
        if not self.connections:
            return
            
        # Convert the message to JSON using to_json method
        try:
            msg_json = message.to_json()
            
            for websocket in self.connections:
                try:
                    await websocket.send(msg_json)
                except Exception as e:
                    print(f"Error sending broadcast to client: {e}")
                    continue
        except Exception as e:
            print(f"Error preparing broadcast message: {e}") 

    def _is_question(self, text: str) -> bool:
        """
        Determine if the given text is a question that requires a user response.
        
        Returns True if the text appears to be a question requiring an answer.
        """
        if not text:
            return False
            
        # Check for explicit question marks
        if '?' in text:
            return True
            
        # Check for question words at the beginning of the text or sentences
        question_starters = [
            "what", "why", "how", "when", "where", "who", "which", 
            "can", "could", "would", "will", "shall", "should", "do", "does", 
            "did", "is", "are", "was", "were", "have", "has", "had"
        ]
        
        # Split into sentences to check if any sentence starts with a question word
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        for sentence in sentences:
            first_word = sentence.split()[0].lower() if sentence.split() else ""
            if first_word in question_starters:
                return True
        
        return False 

    async def text_to_speech_streaming(self, text: str, websocket, is_hangup_message=False) -> bool:
        """Convert text to speech using TTS engine and stream to client in real-time
        
        This method processes the text sentence by sentence and streams audio to the client
        as soon as each sentence is converted to speech, without waiting for the entire text
        to be processed.
        
        Args:
            text: Text to convert to speech
            websocket: WebSocket connection to send audio to
            is_hangup_message: Whether this is a goodbye message before hanging up
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not text or text.strip() == "":
                print("Warning: Empty text provided to streaming TTS, skipping synthesis")
                return False
                
            print(f"Streaming TTS request: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Check if TTS engine is initialized
            if not self.tts_engine or not self.tts_engine.is_initialized():
                print("ERROR: TTS engine not initialized for streaming!")
                return False
                
            # Check if we have a valid websocket and it's still connected
            if websocket not in self.connections:
                print("ERROR: Invalid websocket connection for streaming TTS")
                return False
            
            # Process sentences/segments into audio chunks
            # We need to know when we've reached the end for proper hangup handling
            sentences = self.tts_engine._preprocess_text(text)
            if not sentences:
                print("WARNING: No sentences to process")
                return False
                
            print(f"Processing {len(sentences)} text segments for streaming TTS")
                
            # Process text through streaming TTS - handling each sentence
            chunk_count = 0
            total_bytes = 0
            start_time = time.time()
            first_chunk_time = None
            
            # Process each sentence and convert to audio
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                    
                # Track if this is the last sentence (for hangup handling)
                is_last_sentence = (i == len(sentences) - 1)
                print(f"Processing sentence {i+1}/{len(sentences)}: '{sentence[:30]}{'...' if len(sentence) > 30 else ''}'")
                
                # Generate audio for this sentence
                audio_data = await self.tts_engine._generate_audio(sentence)
                
                if not audio_data:
                    print(f"WARNING: No audio generated for sentence {i+1}")
                    continue
                    
                chunk_count += 1
                is_first_chunk = (chunk_count == 1)
                
                # Only mark as hangup if this is the last sentence AND this is a hangup message
                current_is_hangup = is_hangup_message and is_last_sentence
                
                # Record timing for first chunk
                if is_first_chunk:
                    first_chunk_time = time.time()
                    print(f"First audio chunk ({len(audio_data)} bytes) generated in {first_chunk_time - start_time:.2f}s")
                
                # Send this chunk to the client
                success = await self.send_audio_playback(
                    websocket, 
                    audio_data,
                    is_hangup_message=current_is_hangup,
                    is_final=is_last_sentence
                )
                
                if not success:
                    print(f"ERROR: Failed to send audio for sentence {i+1}, aborting stream")
                    return False
                
                total_bytes += len(audio_data)
                
                # Print stats for first chunk
                if is_first_chunk:
                    print(f"First audio chunk sent in {time.time() - start_time:.2f}s")
                

            
            # Report statistics
            total_time = time.time() - start_time
            print(f"TTS streaming complete: {chunk_count} chunks, {total_bytes/1024:.1f}KB in {total_time:.2f}s")
            if first_chunk_time:
                print(f"Time to first chunk: {first_chunk_time - start_time:.2f}s")
                if chunk_count > 0:
                    print(f"Average chunk size: {total_bytes/chunk_count:.1f} bytes")
                    
            return True
            
        except Exception as e:
            print(f"Error in streaming TTS: {e}")
            import traceback
            traceback.print_exc()
            return False 