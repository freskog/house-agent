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
    model_name: str = "medium-q5_0"
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = None
    task: str = "transcribe"
    initial_prompt: Optional[str] = None
    beam_size: int = 5
    patience: float = 1.0
    use_coreml: bool = True
    
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
    
    @traceable(run_type="chain", name="Audio_Transcription")
    def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000, thread_id: Optional[str] = None) -> TranscriptionResult:
        """
        Transcribe audio data to text directly from memory
        
        Args:
            audio_data: Raw audio data as bytes (assumed to be 16-bit, mono PCM)
            sample_rate: Sample rate of the audio in Hz (default: 16000 Hz)
                         Note: This must match the actual sample rate of the provided audio data
            thread_id: Optional thread ID for LangSmith tracing
        """
        try:
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
                    language="en"
                )
            
            if self.model is None:
                if not self.initialize():
                    raise RuntimeError("Failed to initialize transcription model")
                print("Transcription model initialized on demand")
                
            # Calculate audio duration directly from PCM data (assuming 16-bit audio)
            bytes_per_sample = 2  # 16-bit audio = 2 bytes per sample
            channels = 1  # Mono
            duration = len(audio_data) / (bytes_per_sample * channels * sample_rate)
            
            # Check if audio has content
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
            
            # Apply gain if audio is too quiet but not silent - IMPORTANT FOR QUALITY
            # This block actually improves transcription quality significantly
            if 10.0 < audio_rms < 200.0:
                # Calculate a reasonable gain factor (amplify quiet audio)
                target_rms = 2000.0  # Target a moderate RMS level
                gain_factor = min(target_rms / audio_rms, 15.0)  # Cap gain to avoid excessive amplification
                
                # Apply gain (being careful to avoid integer overflow)
                audio_np_float = audio_np.astype(np.float32)
                audio_np_float = audio_np_float * gain_factor
                
                # Clip to int16 range
                audio_np_float = np.clip(audio_np_float, -32768, 32767)
                audio_np = audio_np_float.astype(np.int16)
                
                # Replace the original audio data
                audio_data = audio_np.tobytes()
                
                # Recalculate RMS after gain
                audio_rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
                print(f"Amplified quiet audio: gain={gain_factor:.1f}x, new RMS={audio_rms:.2f}")
            
            # Check if audio is too quiet - provide a fallback
            if audio_rms < 100:  # Arbitrary threshold, adjust as needed
                print(f"Audio too quiet (RMS: {audio_rms:.2f}), using fallback")
                return TranscriptionResult(
                    text="I detected speech but it was too quiet to transcribe",
                    language="en",
                    duration=duration,
                    process_time=0.0
                )
            
            print(f"Transcribing {len(audio_data)/1024:.1f}KB audio ({duration:.2f}s, RMS: {audio_rms:.2f})")
            
            # Validate audio data
            if len(audio_data) < 1000:
                print(f"Warning: Audio data very short ({len(audio_data)} bytes)")
            
            # Verify CoreML settings
            coreml_setting = os.environ.get("WHISPER_COREML", "Not set")
            print(f"CoreML setting: WHISPER_COREML={coreml_setting}")
            
            # Save to a temporary file - pywhispercpp requires a file path
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                temp_path = temp_file.name
                
                # Write audio data to temporary file
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)
                
                print(f"Temporary file created: {temp_path}")
                
                # Set C locale again before transcription
                os.environ['LC_ALL'] = 'C'
                os.environ['LANG'] = 'C'
                os.environ['LANGUAGE'] = 'C'
                
                # Run transcription
                start_time = time.time()
                print(f"Starting whisper transcription with C locale...")
                segments = self.model.transcribe(temp_path)
                process_time = time.time() - start_time
                
                # Extract result
                segments_list = list(segments)
                text = " ".join(segment.text for segment in segments_list)
                
                # If transcription produced no text, provide a fallback response
                if not text.strip():
                    print(f"Empty transcription after {process_time:.2f}s, using fallback")
                    return TranscriptionResult(
                        text="I could not understand that. Could you please speak more clearly?",
                        language="en",
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
                    language="en",  # Currently hardcoded language
                    segments=segments_formatted,
                    duration=duration,
                    process_time=process_time,
                    thread_id=thread_id
                )
                
                print(f"Transcription completed in {process_time:.2f}s")
                return result
            
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a failure result
            return TranscriptionResult(
                text="Sorry, there was an error processing your speech.",
                language="en",
                thread_id=thread_id
            )
            
    def reset(self):
        """Reset the transcriber state"""
        # No state to reset for now
        pass 