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
        self.last_speech_time = None
        self.recording_start_time = None
        
    def reset(self):
        """Reset the state"""
        self.is_speech_active = False
        self.speech_frames = []
        self.last_speech_time = None
        self.recording_start_time = None
        
    def add_speech_frame(self, frame: bytes):
        """Add a frame to the speech buffer"""
        if not self.is_speech_active:
            self.is_speech_active = True
            self.recording_start_time = time.time()
            
        self.speech_frames.append(frame)
        self.last_speech_time = time.time()
        
    def get_audio_data(self) -> bytes:
        """Get the combined audio data"""
        return b''.join(self.speech_frames)
        
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
                transcription_callback: Optional[TranscriptionCallback] = None):
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
            audio_data = base64.b64decode(audio_data_b64)
            
            # Get client state
            client_state = self.client_states[websocket]
            
            # Occasionally analyze the raw chunk to check for sample rate issues
            if random.random() < 0.05:  # Only analyze ~5% of chunks to reduce overhead
                try:
                    # Convert to numpy for analysis
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    
                except Exception as e:
                    print(f"Error analyzing raw audio chunk: {e}")
                
            # Process through VAD
            vad_result = self.vad_handler.process_chunk(audio_data)
            
            # Check for state transitions and send status updates only on changes
            was_speech_active = client_state.is_speech_active
            
            # Handle speech detection
            current_time = time.time()
            
            if vad_result.is_speech:
                # Add frame to speech buffer
                client_state.add_speech_frame(audio_data)
                
                # Only log on state transitions or occasionally
                if not was_speech_active:
                    print(f"Speech detected - recording started")
                elif len(client_state.speech_frames) % 200 == 0:  # Log less frequently
                    duration = client_state.get_recording_duration()
                    print(f"Recording active - {len(client_state.speech_frames)} frames ({duration:.1f}s)")
                
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
                
                # Only log when approaching the threshold
                if silence_duration > (self.silence_threshold * 0.75) and not hasattr(client_state, '_silence_logged'):
                    print(f"Silence detected, will process in {self.silence_threshold - silence_duration:.1f}s")
                    client_state._silence_logged = True
                
                if silence_duration > self.silence_threshold:
                    print(f"Silence threshold reached, processing recording...")
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
                
            # Get combined audio data
            audio_data = client_state.get_audio_data()
            
            # Define constants for audio format
            SAMPLE_RATE = 16000  # 16kHz - client recording rate
            BYTES_PER_SAMPLE = 2  # 16-bit audio = 2 bytes per sample
            CHANNELS = 1  # Mono
            
            # Validate audio quality
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_duration = len(audio_np) / SAMPLE_RATE
                        
            # Calculate audio metrics
            if len(audio_np) > 0:
                audio_max = np.max(np.abs(audio_np))
                audio_mean = np.mean(np.abs(audio_np))
                audio_rms = np.sqrt(np.mean(np.power(audio_np.astype(np.float32), 2)))
                
                print(f"Audio metrics: max={audio_max}, mean={audio_mean:.2f}, RMS={audio_rms:.2f}")
                
                # Check for potential audio quality issues
                if audio_max < 1000:
                    print("WARNING: Audio level very low, may affect transcription quality")
                if audio_rms < 500:
                    print("WARNING: Audio RMS very low, speech may be too quiet")
            
            # Debug info about audio data
            audio_size = len(audio_data)
            print(f"Processing audio: {audio_size} bytes, ~{audio_duration:.2f} seconds (16kHz, 16-bit mono)")
            
            # Analyze frequency content to validate sample rate
            if len(audio_np) > 1000:  # Need enough samples for frequency analysis
                try:
                    from scipy import signal
                    # Calculate power spectral density
                    f, psd = signal.welch(audio_np, fs=SAMPLE_RATE, nperseg=min(1024, len(audio_np)))
                    
                    # Find frequency with highest energy
                    peak_freq = f[np.argmax(psd)]
                    
                    # Check if most of the energy is below Nyquist frequency (half the sample rate)
                    nyquist = SAMPLE_RATE / 2
                    energy_below_nyquist = np.sum(psd[f < nyquist/2]) / np.sum(psd)
                    
                    print(f"Audio frequency stats: peak={peak_freq:.1f}Hz, energy below {nyquist/2:.1f}Hz: {energy_below_nyquist:.1%}")
                    
                    # If most energy is in very low frequencies, could indicate wrong sample rate
                    if energy_below_nyquist < 0.5:
                        print("WARNING: Unusual frequency distribution - check sample rate")
                except Exception as e:
                    print(f"Error analyzing frequency content: {e}")
            
            
            # Reset client state - this will set is_speech_active to False
            client_state.reset()
            
            # Create a task to run transcription so it doesn't block the WebSocket loop
            loop = asyncio.get_running_loop()
            try:
                print(f"Starting transcription...")
                transcription = await loop.run_in_executor(
                    None,
                    lambda: self.transcriber.transcribe_audio(audio_data)
                )
                print(f"Transcription completed: '{transcription.text}'")
                print(f"Transcription time: {transcription.process_time:.2f}s")
                
                # Debug segments info
                if transcription.segments:
                    print(f"Segments: {len(transcription.segments)}")
                    for i, seg in enumerate(transcription.segments[:2]):  # Print first two segments
                        print(f"  Segment {i}: '{seg['text']}'")
                else:
                    print("No segments in transcription result")
            except Exception as e:
                print(f"Error during transcription: {e}")
                transcription = TranscriptionResult(
                    text="Error transcribing audio",
                    language="en"
                )
            
            # Send transcription to client for transparency
            await self.send_transcription(websocket, transcription)
            
            # Process through the callback
            print(f"Processing transcription: {transcription.text}")
            response = await self.transcription_callback(transcription)
            
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
        
    async def send_audio_playback(self, websocket, audio_data: bytes):
        """Send audio data to client for playback"""
        try:
            # Debug info about the audio data
            print(f"Sending audio playback to client: {len(audio_data)} bytes")
            
            if len(audio_data) == 0:
                print("WARNING: Attempting to send empty audio data")
                return
                
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