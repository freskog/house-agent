"""
Transcription module for processing audio into text.
"""

import os
import tempfile
import wave
import time
import threading
import platform
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import numpy as np

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
                    
                # Set CoreML usage for Apple Silicon if enabled
                if (self.config.use_coreml and platform.system() == "Darwin" 
                    and platform.machine() == "arm64"):
                    os.environ["WHISPER_COREML"] = "1"
                
                # Import here to allow the package to load even without pywhispercpp
                from pywhispercpp.model import Model
                
                print(f"Loading transcription model: {self.config.model_name}")
                self.model = Model(model=self.config.model_name)
                print(f"Transcription model loaded successfully")
                return True
        except Exception as e:
            print(f"Error initializing transcription model: {e}")
            return False
    
    def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        """
        Transcribe audio data to text directly from memory
        
        Args:
            audio_data: Raw audio data as bytes (assumed to be 16-bit, mono PCM)
            sample_rate: Sample rate of the audio in Hz (default: 16000 Hz)
                         Note: This must match the actual sample rate of the provided audio data
        """
        if self.model is None:
            raise RuntimeError("Transcription model not initialized")
                
        try:
            # Calculate audio duration directly from PCM data (assuming 16-bit audio)
            bytes_per_sample = 2  # 16-bit audio = 2 bytes per sample
            channels = 1  # Mono
            duration = len(audio_data) / (bytes_per_sample * channels * sample_rate)
            
            # Check if audio has content
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
            
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
            
            # Save to a temporary file - pywhispercpp requires a file path
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                temp_path = temp_file.name
                
                # Write audio data to temporary file
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data)
                
                # Run transcription
                start_time = time.time()
                segments = self.model.transcribe(temp_path)
                process_time = time.time() - start_time
                
                # Extract result
                segments_list = list(segments)
                text = " ".join(segment.text for segment in segments_list)
                
                # If transcription produced no text, provide a fallback response
                if not text.strip():
                    print(f"Empty transcription after {process_time:.2f}s, using fallback")
                    
                    # Use audio characteristics to determine a response
                    if duration < 1.0:
                        fallback_text = "I heard a brief sound"
                    elif duration < 3.0:
                        fallback_text = "I heard you say something short"
                    else:
                        fallback_text = "I heard you speaking but couldn't understand what you said"
                        
                    return TranscriptionResult(
                        text=fallback_text,
                        language="en",
                        duration=duration,
                        process_time=process_time
                    )
                
                print(f"Transcription result ({process_time:.2f}s): '{text}'")
                
                # Create result object
                return TranscriptionResult(
                    text=text,
                    language="en",  # pywhispercpp doesn't expose language info in the same way
                    segments=[{
                        "id": i,
                        "text": segment.text,
                        "start": segment.start_time if hasattr(segment, 'start_time') else 0.0,
                        "end": segment.end_time if hasattr(segment, 'end_time') else 0.0
                    } for i, segment in enumerate(segments_list)],
                    duration=duration,
                    process_time=process_time
                )
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            import traceback
            traceback.print_exc()
            return TranscriptionResult(
                text=f"Error transcribing audio: {e}",
                language="en"
            )
            
    def reset(self):
        """Reset the transcriber state"""
        # No state to reset for now
        pass 