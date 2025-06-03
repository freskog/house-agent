"""
Transcription module for processing audio into text.
"""

import os
import tempfile
import wave
import time
import threading
import platform
import locale
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import numpy as np
from langsmith import traceable

class TranscriptionConfig(BaseModel):
    """Configuration for transcription"""
    model_name: str = "large-v3-turbo"
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = None
    task: str = "transcribe"
    initial_prompt: Optional[str] = None
    beam_size: int = 5
    patience: float = 1.0
    use_coreml: bool = True
    use_langsmith: bool = False  # Disabled by default
    
class TranscriptionResult(BaseModel):
    """Result from transcription"""
    text: str
    language: str
    segments: List[Dict[str, Any]] = []
    duration: float = 0.0
    process_time: float = 0.0
    thread_id: Optional[str] = None
    confidence: float = 1.0
    # Note: client reference is now handled via a non-serialized _websocket attribute
    
class Transcriber:
    """Handles transcription of audio data"""
    
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()
        self.model = None
        self.lock = threading.Lock()
        # Initialize eagerly
        print("Initializing transcription model...")
        self.initialize()
        
    def initialize(self) -> bool:
        """Initialize the transcription model synchronously"""
        if self.model is not None:
            return True
            
        try:
            with self.lock:
                if self.model is not None:  # Double-check under lock
                    return True
                
                # CRITICAL: Fix locale before initializing Whisper to prevent segfault
                # This must be done right before Whisper initialization
                print("Setting C locale to avoid Whisper segmentation fault...")
                try:
                    # Store original locale to restore later if needed
                    original_locale = locale.getlocale()
                    # Set locale to C
                    locale.setlocale(locale.LC_ALL, 'C')
                    # Also set environment variables
                    os.environ['LC_ALL'] = 'C'
                    os.environ['LANG'] = 'C'
                    os.environ['LANGUAGE'] = 'C'
                    print("Locale set to C")
                except Exception as e:
                    print(f"Warning: Failed to set locale: {e}")
                
                # Set CoreML usage for Apple Silicon if enabled
                if (self.config.use_coreml and platform.system() == "Darwin" 
                    and platform.machine() == "arm64"):
                    os.environ["WHISPER_COREML"] = "1"
                
                # Import here to allow the package to load even without pywhispercpp
                from pywhispercpp.model import Model
                
                print(f"Loading transcription model: {self.config.model_name}")
                self.model = Model(model=self.config.model_name)
                print(f"Transcription model loaded successfully")
                
                # Restore original locale if needed
                try:
                    if original_locale:
                        locale.setlocale(locale.LC_ALL, original_locale)
                except Exception:
                    # If restoring fails, leave as C - operation succeeded anyway
                    pass
                
                return True
        except Exception as e:
            print(f"Error initializing transcription model: {e}")
            return False
    
    def _convert_audio_to_numpy(self, audio_data: bytes, sample_rate: int = 16000) -> np.ndarray:
        """Convert raw audio bytes to normalized numpy array for Whisper
        
        Args:
            audio_data: Raw audio data as bytes (16-bit, mono PCM)
            sample_rate: Sample rate of the audio
            
        Returns:
            Normalized float32 numpy array suitable for Whisper
        """
        # Convert bytes to int16 numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # Normalize to float32 in the range [-1.0, 1.0] as expected by Whisper
        # Whisper expects audio normalized to [-1, 1] range
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        return audio_float
    
    def _apply_audio_gain_optimization(self, audio_np: np.ndarray) -> tuple[np.ndarray, float]:
        """Apply gain optimization to improve transcription quality for quiet audio
        
        Args:
            audio_np: Original audio array (int16)
            
        Returns:
            Tuple of (optimized_audio_bytes, applied_gain_factor)
        """
        # Calculate RMS for volume detection
        audio_rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
        
        # Apply gain if audio is too quiet but not silent - IMPORTANT FOR QUALITY
        if 10.0 < audio_rms < 200.0:
            # Calculate a reasonable gain factor (amplify quiet audio)
            target_rms = 2000.0  # Target a moderate RMS level
            gain_factor = min(target_rms / audio_rms, 15.0)  # Cap gain to avoid excessive amplification
            
            # Apply gain (being careful to avoid integer overflow)
            audio_np_float = audio_np.astype(np.float32)
            audio_np_float = audio_np_float * gain_factor
            
            # Clip to int16 range
            audio_np_float = np.clip(audio_np_float, -32768, 32767)
            audio_np_optimized = audio_np_float.astype(np.int16)
            
            # Recalculate RMS after gain
            new_rms = np.sqrt(np.mean(audio_np_optimized.astype(np.float32)**2))
            print(f"Amplified quiet audio: gain={gain_factor:.1f}x, RMS {audio_rms:.2f} -> {new_rms:.2f}")
            
            return audio_np_optimized.tobytes(), gain_factor
        else:
            return audio_np.tobytes(), 1.0

    @traceable(run_type="chain", name="Audio_Transcription", skip_if=lambda self: not self.config.use_langsmith)
    def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000, thread_id: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe audio data to text using optimized in-memory processing
        
        Args:
            audio_data: Raw audio data as bytes (assumed to be 16-bit, mono PCM)
            sample_rate: Sample rate of the audio in Hz (default: 16000 Hz)
                         Note: This must match the actual sample rate of the provided audio data
            thread_id: Optional thread ID for LangSmith tracing
        """
        try:
            # Only configure LangSmith if it's enabled
            if self.config.use_langsmith:
                # Configure environment for LangSmith if using a non-standard API key format
                if os.environ.get("LANGSMITH_API_KEY", "").startswith("lsv2_"):
                    os.environ["LANGSMITH_ALLOW_ANY_API_KEY_FORMAT"] = "true"
                
                # Set thread_id in environment if provided
                if thread_id:
                    os.environ["LANGSMITH_THREAD_ID"] = thread_id
                
            # Validate audio data
            if not audio_data:
                print("ERROR: Received empty audio data")
                return TranscriptionResult(
                    text="",
                    language=self.config.language or "auto"
                )
            
            if self.model is None:
                if not self.initialize():
                    raise RuntimeError("Failed to initialize transcription model")
                print("Transcription model initialized on demand")
                
            # Calculate audio duration directly from PCM data (assuming 16-bit audio)
            bytes_per_sample = 2  # 16-bit audio = 2 bytes per sample
            channels = 1  # Mono
            duration = len(audio_data) / (bytes_per_sample * channels * sample_rate)
            
            # Convert to numpy and check if audio has content
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
            
            # Check if audio is too quiet - provide a fallback
            if audio_rms < 100:  # Arbitrary threshold, adjust as needed
                print(f"Audio too quiet (RMS: {audio_rms:.2f}), using fallback")
                return TranscriptionResult(
                    text="I detected speech but it was too quiet to transcribe",
                    language=self.config.language or "auto",
                    duration=duration,
                    process_time=0.0
                )
            
            print(f"Transcribing {len(audio_data)/1024:.1f}KB audio ({duration:.2f}s, RMS: {audio_rms:.2f})")
            
            # Validate audio data
            if len(audio_data) < 1000:
                print(f"Warning: Audio data very short ({len(audio_data)} bytes)")
            
            # Apply gain optimization if needed
            optimized_audio_data, gain_factor = self._apply_audio_gain_optimization(audio_np)
            
            # Convert optimized audio to normalized numpy array for Whisper
            audio_float = self._convert_audio_to_numpy(optimized_audio_data, sample_rate)
            
            # Verify CoreML settings
            coreml_setting = os.environ.get("WHISPER_COREML", "Not set")
            print(f"CoreML setting: WHISPER_COREML={coreml_setting}")
            
            # Set C locale again before transcription
            os.environ['LC_ALL'] = 'C'
            os.environ['LANG'] = 'C'
            os.environ['LANGUAGE'] = 'C'
            
            # Run transcription using optimized in-memory processing
            start_time = time.time()
            print(f"Starting whisper transcription with in-memory processing...")
            
            # Use numpy array directly - this eliminates file I/O completely!
            # Pass translate=False to ensure we keep the original language
            segments = self.model.transcribe(
                audio_float,
                translate=False,
                language=self.config.language or "auto"
            )
            process_time = time.time() - start_time
            
            # Extract result
            segments_list = list(segments)
            text = " ".join(segment.text for segment in segments_list)
            
            # Get the detected language from the first segment
            detected_language = None
            if segments_list and hasattr(segments_list[0], 'language'):
                detected_language = segments_list[0].language
            
            # If transcription produced no text, provide a fallback response
            if not text.strip():
                print(f"Empty transcription after {process_time:.2f}s, using fallback")
                return TranscriptionResult(
                    text="I could not understand that. Could you please speak more clearly?",
                    language=detected_language or self.config.language or "auto",
                    duration=duration,
                    process_time=process_time,
                    thread_id=thread_id
                )
                
            # Create segments array in the expected format
            segments_formatted = []
            for i, segment in enumerate(segments_list):
                segments_formatted.append({
                    "id": i,
                    "text": segment.text,
                    "start": segment.start_time if hasattr(segment, 'start_time') else 0.0,
                    "end": segment.end_time if hasattr(segment, 'end_time') else 0.0
                })
            
            # Create and return the result
            result = TranscriptionResult(
                text=text.strip(),
                language=detected_language or self.config.language or "auto",
                segments=segments_formatted,
                duration=duration,
                process_time=process_time,
                thread_id=thread_id
            )
            
            print(f"Transcription completed in {process_time:.2f}s (gain: {gain_factor:.1f}x)")
            return result
            
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: try the old file-based method if numpy method fails
            print("Attempting fallback to file-based transcription...")
            return self._transcribe_fallback_file_method(audio_data, sample_rate, thread_id)
            
    def _transcribe_fallback_file_method(self, audio_data: bytes, sample_rate: int, thread_id: Optional[str]) -> TranscriptionResult:
        """Fallback file-based transcription method in case numpy method fails"""
        try:
            # Calculate audio duration
            duration = len(audio_data) / (2 * sample_rate)  # 16-bit = 2 bytes per sample
            
            # Save to a temporary file - pywhispercpp requires a file path
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                temp_path = temp_file.name
                
                # Write audio data to temporary file
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)
                
                print(f"Fallback: Using temporary file {temp_path}")
                
                # Run transcription
                start_time = time.time()
                segments = self.model.transcribe(
                    temp_path,
                    translate=False,
                    language=self.config.language or "auto"
                )
                process_time = time.time() - start_time
                
                # Extract result
                segments_list = list(segments)
                text = " ".join(segment.text for segment in segments_list)
                
                if not text.strip():
                    text = "I could not understand that. Could you please speak more clearly?"
                
                segments_formatted = []
                for i, segment in enumerate(segments_list):
                    segments_formatted.append({
                        "id": i,
                        "text": segment.text,
                        "start": segment.start_time if hasattr(segment, 'start_time') else 0.0,
                        "end": segment.end_time if hasattr(segment, 'end_time') else 0.0
                    })
                
                return TranscriptionResult(
                    text=text.strip(),
                    language="en",
                    segments=segments_formatted,
                    duration=duration,
                    process_time=process_time,
                    thread_id=thread_id
                )
                
        except Exception as fallback_error:
            print(f"Fallback transcription also failed: {fallback_error}")
            return TranscriptionResult(
                text="Sorry, there was an error processing your speech.",
                language="en",
                thread_id=thread_id
            )
            
    def reset(self):
        """Reset the transcriber state"""
        # No state to reset for now
        pass 