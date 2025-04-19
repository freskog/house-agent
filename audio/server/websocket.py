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
from typing import Optional, Dict, Any, Callable, Set, List, Awaitable, Union
from .protocol import (
    Message, MessageType, AudioConfig, DEFAULT_AUDIO_CONFIG,
    AudioStreamPayload, AudioPlaybackPayload, StatusPayload, ErrorPayload
)
from .vad import VADHandler, VADConfig, VADResult
from .transcribe import Transcriber, TranscriptionConfig, TranscriptionResult
from .tts import TTSEngine, TTSConfig

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
        
    def reset(self):
        """Reset the state"""
        self.is_speech_active = False
        # Keep speech frames for debugging - they'll be replaced on next speech detection
        self.last_speech_time = time.time()
        self.recording_start_time = time.time()
        # Don't clear pre_vad_buffer - we want to keep recent audio
        
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
        self.vad_handler = VADHandler(config=vad_config)
        self.transcriber = Transcriber(config=transcription_config)
        self.tts_engine = TTSEngine(config=tts_config)
        self.connections: Set[websockets.WebSocketServerProtocol] = set()
        self.client_states: Dict[websockets.WebSocketServerProtocol, AudioProcessingState] = {}
        self.sequence = 0
        self.silence_threshold = 1.0  # seconds of silence before processing
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
        # Start server
        self.server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port
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
                    else:
                        print(f"Unhandled message type: {msg.type}")
                        
                except json.JSONDecodeError:
                    print("Invalid message format")
                except Exception as e:
                    print(f"Error handling message: {e}")
                    await self.send_error(websocket, str(e))
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"Connection closed from {websocket.remote_address}")
        finally:
            if websocket in self.connections:
                self.connections.remove(websocket)
            if websocket in self.client_states:
                del self.client_states[websocket]
            
    async def handle_audio_stream(self, websocket, message: Message):
        """Handle incoming audio stream data"""
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
                    
                # Verify audio data format and content immediately (only for debugging, can be disabled)
                # Commented out to reduce logging and improve performance
                # try:
                #     audio_np = np.frombuffer(audio_data, dtype=np.int16)
                #     audio_rms = np.sqrt(np.mean(np.power(audio_np.astype(np.float32), 2)))
                #     audio_max = np.max(np.abs(audio_np)) if len(audio_np) > 0 else 0
                #     
                #     # Log audio stats periodically
                #     if random.random() < 0.05:  # Log ~5% of frames
                #         print(f"Audio frame: size={len(audio_data)} bytes, RMS={audio_rms:.1f}, Max={audio_max}")
                #     
                #     # Check for extremely quiet audio
                #     if audio_rms < 5.0:
                #         if random.random() < 0.1:  # Don't log every frame to avoid spam
                #             print(f"Very quiet audio frame received: RMS={audio_rms:.1f}")
                # except Exception as e:
                #     print(f"Error analyzing audio frame: {e}")
                    
            except Exception as e:
                print(f"ERROR: Failed to decode base64 audio data: {e}")
                await self.send_error(websocket, f"Invalid audio data format: {e}")
                return
            
            # Get client state
            client_state = self.client_states[websocket]
            
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
                    
                    # Log the pre-VAD buffer sizes
                    buffer_sizes = [len(frame) for frame in client_state.pre_vad_buffer]
                    print(f"Speech detected - adding pre-buffer: {len(client_state.pre_vad_buffer)} frames, sizes: {buffer_sizes}")
                    
                    # Add all non-empty frames from pre-VAD buffer
                    for frame in client_state.pre_vad_buffer:
                        if frame and len(frame) > 0:
                            client_state.speech_frames.append(frame)
                        else:
                            print("WARNING: Skipping empty frame in pre-VAD buffer")
                            
                    print(f"Added {len(client_state.speech_frames)} frames from pre-buffer")
                
                # Always add the current frame to speech collection
                if audio_data and len(audio_data) > 0:
                    client_state.speech_frames.append(audio_data)
                    client_state.last_speech_time = current_time
                    
                    # Periodically log the total collected frames (reduced frequency)
                    if len(client_state.speech_frames) % 50 == 0:
                        print(f"Recording: {len(client_state.speech_frames)} frames ({len(client_state.speech_frames)*2/16000:.1f}s)")
                
                # Debug audio data before processing - DISABLED for performance
                # try:
                #     audio_np = np.frombuffer(audio_data, dtype=np.int16)
                #     if len(audio_np) > 0:
                #         audio_rms = np.sqrt(np.mean(np.power(audio_np.astype(np.float32), 2)))
                #         audio_max = np.max(np.abs(audio_np))
                #         print(f"DEBUG AUDIO: Frame size={len(audio_data)} bytes, RMS={audio_rms:.1f}, Max={audio_max}")
                #         
                #         # Check for zero or near-zero audio
                #         if audio_rms < 50:
                #             print("WARNING: Very low audio level detected in this frame")
                #         if audio_max < 100:
                #             print("WARNING: Very low peak amplitude in this frame")
                #             
                #         # Periodically save individual frames for inspection - DISABLED for performance 
                #         # if random.random() < 0.02:  # Save ~2% of frames
                #         #    os.makedirs("frame_samples", exist_ok=True)
                #         #    frame_path = f"frame_samples/frame_{int(time.time())}_{len(audio_data)}.wav"
                #         #    with wave.open(frame_path, 'wb') as wf:
                #         #        wf.setnchannels(1)  # Mono
                #         #        wf.setsampwidth(2)  # 16-bit
                #         #        wf.setframerate(16000)  # 16kHz
                #         #        wf.writeframes(audio_data)
                #         #    print(f"Saved sample frame to {frame_path}")
                # except Exception as e:
                #     print(f"Error analyzing audio frame: {e}")
                
                # Only log on state transitions or occasionally
                if not was_speech_active:
                    print(f"Speech detected - recording started")
                # Removed redundant logging to reduce console output
                
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
                
                # Only log when silence is first detected
                if silence_duration > (self.silence_threshold * 0.5) and not hasattr(client_state, '_silence_logged'):
                    print(f"Silence detected, will process soon...")
                    client_state._silence_logged = True
                
                if silence_duration > self.silence_threshold:
                    print(f"Processing after {silence_duration:.1f}s silence")
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
            
        except Exception as e:
            print(f"Error processing audio stream: {e}")
            await self.send_error(websocket, str(e))
            
    async def process_recording(self, websocket, client_state: AudioProcessingState):
        """Process the recorded audio"""
        try:
            if not client_state.speech_frames:
                print("No speech frames to process!")
                return
                
            # Debug: Log speech frames summary (minimal logging)
            frame_count = len(client_state.speech_frames)
            if frame_count > 0:
                frame_sizes = [len(frame) for frame in client_state.speech_frames]
                avg_size = sum(frame_sizes)/len(frame_sizes) if frame_sizes else 0
                print(f"Processing {frame_count} speech frames (~{frame_count*avg_size/1024:.1f}KB)")
                
            # Save raw speech frames directly for debugging - DISABLED for performance
            # This block is useful for debugging but adds latency to normal operation
            # try:
            #     os.makedirs("speech_frames", exist_ok=True)
            #     timestamp = int(time.time())
            #     
            #     # Individually save a few frames as samples
            #     for i, frame in enumerate(client_state.speech_frames[:3]):  # Save first 3 frames
            #         if frame and len(frame) > 0:
            #             frame_path = f"speech_frames/frame_{timestamp}_{i}.wav"
            #             with wave.open(frame_path, 'wb') as wf:
            #                 wf.setnchannels(CHANNELS)
            #                 wf.setsampwidth(BYTES_PER_SAMPLE)
            #                 wf.setframerate(SAMPLE_RATE)
            #                 wf.writeframes(frame)
            #     
            #     # Also save all frames as a single file directly (before the get_audio_data call)
            #     all_frames_data = b''.join(client_state.speech_frames)
            #     frames_path = f"speech_frames/all_frames_{timestamp}.wav"
            #     with wave.open(frames_path, 'wb') as wf:
            #         wf.setnchannels(CHANNELS)
            #         wf.setsampwidth(BYTES_PER_SAMPLE)
            #         wf.setframerate(SAMPLE_RATE)
            #         wf.writeframes(all_frames_data)
            #     print(f"Saved all speech frames to {frames_path}")
            # except Exception as e:
            #     print(f"Error saving speech frames: {e}")
                
            # Get combined audio data
            audio_data = client_state.get_audio_data()
            
            # Immediate check for empty audio data
            if not audio_data or len(audio_data) == 0:
                print("ERROR: Empty audio data after combining frames!")
                # Try to recover using individual frames
                if client_state.speech_frames and any(len(f) > 0 for f in client_state.speech_frames):
                    print("Attempting to recover using last non-empty frame...")
                    # Find the last non-empty frame
                    for frame in reversed(client_state.speech_frames):
                        if len(frame) > 0:
                            audio_data = frame
                            print(f"Using frame with {len(audio_data)} bytes")
                            break
                if not audio_data or len(audio_data) == 0:
                    print("ERROR: Could not recover any audio data, aborting transcription")
                    return
            
            # Define constants for audio format
            SAMPLE_RATE = 16000  # 16kHz - client recording rate
            BYTES_PER_SAMPLE = 2  # 16-bit audio = 2 bytes per sample
            CHANNELS = 1  # Mono
            
            # Validate audio quality - minimal validation to improve performance
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_duration = len(audio_np) / SAMPLE_RATE
            
            # Only calculate audio metrics if really needed - MINIMAL version
            audio_size = len(audio_data)
            print(f"Processing audio: {audio_size/1024:.1f}KB, ~{audio_duration:.2f}s")
            
            # Save debug recordings - REMOVED in favor of the flag-based recording
            # Now handled after transcription with proper naming
            
            # Reset client state - this will set is_speech_active to False
            client_state.reset()
            
            # Reset transcriber to ensure it's in a fresh state
            self.transcriber.reset()
            
            # Create a task to run transcription so it doesn't block the WebSocket loop
            loop = asyncio.get_running_loop()
            try:
                # Using run_in_executor for CPU-bound operation
                print(f"Starting transcription...")
                transcription_start = time.time()
                transcription = await loop.run_in_executor(
                    None,
                    lambda: self.transcriber.transcribe_audio(audio_data)
                )
                transcription_time = time.time() - transcription_start
                print(f"Transcription completed in {transcription_time:.2f}s: '{transcription.text}'")
                
                # Add client reference but as a non-serialized attribute (with leading underscore)
                # This prevents it from being included in JSON serialization
                setattr(transcription, '_websocket', websocket)
                
                # Debug segments info - DISABLED for cleaner logs
                # if transcription.segments:
                #    print(f"Segments: {len(transcription.segments)}")
                #    for i, seg in enumerate(transcription.segments[:2]):  # Print first two segments
                #        print(f"  Segment {i}: '{seg['text']}'")
                # else:
                #    print("No segments in transcription result")
            except Exception as e:
                print(f"Error during transcription: {e}")
                transcription = TranscriptionResult(
                    text="Error transcribing audio",
                    language="en"
                )
            
            # Send transcription to client for transparency
            await self.send_transcription(websocket, transcription)
            
            # Save recording if the flag is enabled and we have valid transcription
            if self.save_recordings and transcription.text and audio_data:
                try:
                    # Create recordings directory if it doesn't exist
                    os.makedirs("recordings", exist_ok=True)
                    
                    # Use the first 15 characters of transcription for filename
                    # Replace invalid filename characters with underscores
                    base_name = transcription.text[:15].strip()
                    safe_name = "".join([c if c.isalnum() or c in " .,_-" else "_" for c in base_name])
                    safe_name = safe_name.replace(" ", "_")
                    
                    # Add timestamp to ensure uniqueness
                    timestamp = int(time.time())
                    filename = f"recordings/{safe_name}_{timestamp}.wav"
                    
                    # Save as WAV file
                    with wave.open(filename, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(BYTES_PER_SAMPLE)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(audio_data)
                    
                    print(f"Saved recording: {filename}")
                except Exception as e:
                    print(f"Error saving recording: {e}")
            
            # Process through the callback
            callback_start = time.time()
            print(f"Processing transcription...")
            response = await self.transcription_callback(transcription)
            callback_time = time.time() - callback_start
            print(f"Callback completed in {callback_time:.2f}s")
            
            # Send a status update to indicate we're back to listening
            await self.send_status(websocket, VADResult(is_speech=False, confidence=0.0), state="listening")
            
            # Handle the response
            if isinstance(response, str):
                # Convert text to audio via TTS
                print(f"Converting response to speech: '{response}'")
                audio_response = await self.text_to_speech(response)
                # Send audio for playback
                await self.send_audio_playback(websocket, audio_response)
            elif isinstance(response, bytes):
                # Raw audio data
                await self.send_audio_playback(websocket, response)
            elif isinstance(response, dict):
                # Handle dictionary response (could contain both text and audio)
                if "text" in response:
                    # Send text response
                    await self.send_transcription(websocket, TranscriptionResult(
                        text=response["text"],
                        language="en"
                    ))
                
                if "audio" in response:
                    # Send audio response
                    await self.send_audio_playback(websocket, response["audio"])
            
        except Exception as e:
            print(f"Error processing recording: {e}")
            await self.send_error(websocket, str(e))

    async def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech using TTS engine"""
        try:
            if not text or text.strip() == "":
                print("Warning: Empty text provided to TTS, skipping synthesis")
                return b''
                
            print(f"TTS request: '{text}'")
            # Use the TTS engine to synthesize speech
            audio_data = await self.tts_engine.synthesize(text)
            
            print(f"TTS response size: {len(audio_data)} bytes")
            if len(audio_data) < 100:
                print("Warning: TTS produced very small or empty audio data")
            
            return audio_data
        except Exception as e:
            print(f"Error in TTS processing: {e}")
            # Return empty audio as fallback
            return b''
            
    async def _default_transcription_callback(self, transcription: TranscriptionResult) -> str:
        """Default callback that just echoes the transcription"""
        return f"I heard: {transcription.text}"
            
    async def send_status(self, websocket, vad_result: VADResult, state: Optional[str] = None):
        """Send status update to client"""
        # Create status payload
        status_payload = StatusPayload(
            is_speech=vad_result.is_speech,
            confidence=vad_result.confidence,
            timestamps=vad_result.timestamps
        )
        
        # Add optional state information
        payload_dict = status_payload.dict()
        if state:
            payload_dict["state"] = state
        
        # Create message
        message = Message(
            type=MessageType.STATUS,
            timestamp=time.time(),
            sequence=self.sequence,
            payload=payload_dict
        )
        self.sequence += 1
        
        # Send message
        await websocket.send(message.json())
        
    async def send_transcription(self, websocket, result: TranscriptionResult):
        """Send transcription result to client"""
        message = Message(
            type=MessageType.STATUS,
            timestamp=time.time(),
            sequence=self.sequence,
            payload={
                "transcription": result.dict()
            }
        )
        self.sequence += 1
        
        # Send message
        await websocket.send(message.json())
        
    async def send_error(self, websocket, error: str, code: Optional[int] = None):
        """Send error message to client"""
        # Create error payload
        error_payload = ErrorPayload(
            error=error,
            code=code
        )
        
        # Create message
        message = Message(
            type=MessageType.ERROR,
            timestamp=time.time(),
            sequence=self.sequence,
            payload=error_payload.dict()
        )
        self.sequence += 1
        
        # Send message
        await websocket.send(message.json())
        
    async def send_audio_playback(self, websocket, audio_data: bytes, is_hangup_message=False):
        """Send audio data to client for playback
        
        Args:
            websocket: The client websocket
            audio_data: The audio data to send
            is_hangup_message: Whether this is a hangup message (optimizes for faster delivery)
        """
        try:
            # Debug info about the audio data
            print(f"Sending audio playback to client: {len(audio_data)} bytes{' (hangup message)' if is_hangup_message else ''}")
            
            if len(audio_data) == 0:
                print("WARNING: Attempting to send empty audio data")
                return
                
            # For hangup messages, prioritize speed over detailed format checking
            if is_hangup_message:
                # Send as a single message for faster delivery
                audio_payload = AudioPlaybackPayload(
                    audio=base64.b64encode(audio_data).decode('utf-8')
                )
                
                # Create message
                message = Message(
                    type=MessageType.AUDIO_PLAYBACK,
                    timestamp=time.time(),
                    sequence=self.sequence,
                    payload=audio_payload.dict()
                )
                self.sequence += 1
                
                # Send message
                await websocket.send(message.json())
                print(f"Hangup audio message sent successfully")
                return
            
            # For regular messages, continue with normal processing
            # Check WAV header if present
            if len(audio_data) >= 44:  # Minimum WAV header size
                header = audio_data[:12]
                header_str = ' '.join([f'{b:02x}' for b in header])
                print(f"Audio header first 12 bytes: {header_str}")
                
                # Try to decode the RIFF header
                try:
                    riff_text = header[:4].decode('ascii')
                    wave_text = audio_data[8:12].decode('ascii')
                    print(f"Header format: {riff_text}/{wave_text}")
                    
                    # Get audio format details from header
                    import struct
                    num_channels = struct.unpack('<H', audio_data[22:24])[0]
                    sample_rate = struct.unpack('<I', audio_data[24:28])[0]
                    bits_per_sample = struct.unpack('<H', audio_data[34:36])[0]
                    print(f"WAV format: {num_channels} channels, {sample_rate} Hz, {bits_per_sample} bits")
                except Exception as e:
                    print(f"Error parsing WAV header: {e}")
            
            # Check if audio data exceeds WebSocket message size limit (900KB to be safe)
            # WebSockets typically have a 1MB limit, so we stay well under that
            MAX_CHUNK_SIZE = 900 * 1024  # 900 KB
            
            if len(audio_data) > MAX_CHUNK_SIZE:
                print(f"Audio data too large ({len(audio_data)} bytes), sending in chunks")
                
                # If it's a WAV file, we need to preserve the header in each chunk
                is_wav = len(audio_data) >= 44 and audio_data[:4] == b'RIFF'
                
                if is_wav:
                    # Extract WAV header (44 bytes) - we'll need to include it in each chunk
                    wav_header = audio_data[:44]
                    audio_content = audio_data[44:]
                    
                    # Determine number of chunks needed
                    chunk_size = MAX_CHUNK_SIZE - 44  # Account for header in each chunk
                    num_chunks = (len(audio_content) + chunk_size - 1) // chunk_size
                    
                    print(f"Splitting WAV into {num_chunks} chunks")
                    
                    # Send each chunk as a separate message
                    for i in range(num_chunks):
                        start = i * chunk_size
                        end = min(start + chunk_size, len(audio_content))
                        
                        # Combine header with this chunk of audio content
                        chunk_data = wav_header + audio_content[start:end]
                        
                        # Create payload with chunk info
                        audio_payload = {
                            "audio": base64.b64encode(chunk_data).decode('utf-8'),
                            "chunk_info": {
                                "index": i,
                                "total": num_chunks,
                                "is_wav": True
                            }
                        }
                        
                        # Create message
                        message = Message(
                            type=MessageType.AUDIO_PLAYBACK,
                            timestamp=time.time(),
                            sequence=self.sequence,
                            payload=audio_payload
                        )
                        self.sequence += 1
                        
                        # Send message with a small delay to prevent overwhelming the socket
                        await websocket.send(message.json())
                        await asyncio.sleep(0.01)  # Small delay between chunks
                        
                    print(f"Successfully sent {num_chunks} WAV chunks")
                    return
                else:
                    # For non-WAV data, just split the raw bytes
                    num_chunks = (len(audio_data) + MAX_CHUNK_SIZE - 1) // MAX_CHUNK_SIZE
                    print(f"Splitting audio data into {num_chunks} chunks")
                    
                    for i in range(num_chunks):
                        start = i * MAX_CHUNK_SIZE
                        end = min(start + MAX_CHUNK_SIZE, len(audio_data))
                        
                        chunk_data = audio_data[start:end]
                        
                        # Create payload with chunk info
                        audio_payload = {
                            "audio": base64.b64encode(chunk_data).decode('utf-8'),
                            "chunk_info": {
                                "index": i, 
                                "total": num_chunks,
                                "is_wav": False
                            }
                        }
                        
                        # Create message
                        message = Message(
                            type=MessageType.AUDIO_PLAYBACK,
                            timestamp=time.time(),
                            sequence=self.sequence,
                            payload=audio_payload
                        )
                        self.sequence += 1
                        
                        # Send message with a small delay to prevent overwhelming the socket
                        await websocket.send(message.json())
                        await asyncio.sleep(0.01)  # Small delay between chunks
                        
                    print(f"Successfully sent {num_chunks} audio chunks")
                    return
            
            # If the audio is small enough, send it as a single message as before
            # Create audio playback payload
            audio_payload = AudioPlaybackPayload(
                audio=base64.b64encode(audio_data).decode('utf-8')
            )
            
            # Create message
            message = Message(
                type=MessageType.AUDIO_PLAYBACK,
                timestamp=time.time(),
                sequence=self.sequence,
                payload=audio_payload.dict()
            )
            self.sequence += 1
            
            # Send message
            await websocket.send(message.json())
            print(f"Audio playback message sent successfully")
        except Exception as e:
            print(f"Error sending audio playback: {e}")
            import traceback
            traceback.print_exc()
            
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
            except Exception as e:
                print(f"Error sending hanging_up status: {e}")
                import traceback
                traceback.print_exc()
                
            # Send a goodbye message if provided
            if message:
                try:
                    print(f"Step 2: Synthesizing goodbye message: '{message}'")
                    # First convert the text to speech
                    audio_data = await self.tts_engine.synthesize(message)
                    
                    # Check audio was generated
                    if not audio_data or len(audio_data) == 0:
                        print("Warning: Failed to generate audio for goodbye message")
                    else:
                        print(f"Step 3: Sending {len(audio_data)} bytes of goodbye audio")
                        # Send the audio with hangup flag for optimized delivery
                        await self.send_audio_playback(websocket, audio_data, is_hangup_message=True)
                        # Very small delay just to ensure the audio message is sent
                        await asyncio.sleep(0.1)
                        print("Goodbye audio sent successfully")
                except Exception as e:
                    print(f"Error sending goodbye message: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Remove the connection from tracking collections immediately
            # This prevents any more messages from being sent to this client
            print(f"Step 4: Removing client from tracking collections")
            if websocket in self.connections:
                self.connections.remove(websocket)
                print("Connection removed from tracking")
            if websocket in self.client_states:
                del self.client_states[websocket]
                print("Client state removed from tracking")
            
            # Close the connection
            print(f"Step 5: Closing connection with {websocket.remote_address}")
            try:
                await websocket.close(code=1000, reason="Call ended")
                print("WebSocket close command sent")
            except Exception as e:
                print(f"Error closing websocket: {e}")
                import traceback
                traceback.print_exc()
                
            print("Hang up completed successfully")
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
            
        msg_json = message.json()
        for websocket in self.connections:
            try:
                await websocket.send(msg_json)
            except:
                continue 