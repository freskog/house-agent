"""
Voice Activity Detection (VAD) module for detecting speech in audio.
"""

import os
import torch
import numpy as np
import time
from typing import Optional, Dict, List, Tuple, Any, Union
import sys

# Cache for models
_vad_cache = {
    "model": None,
    "speech_timestamps_fn": None
}

class VADConfig:
    """Configuration for VAD"""
    def __init__(self, 
                 model_name: str = "silero_vad",
                 threshold: float = 0.4,
                 sample_rate: int = 16000,
                 min_speech_duration: float = 0.25,  # Reduced to catch brief speech
                 min_silence_duration: float = 1.0,  # Seconds of silence before cutting off
                 window_size_samples: int = 512,    # Process in small windows for quick response
                 speech_pad_ms: int = 150,          # Add padding around speech for smoother detection
                 overlap_factor: float = 0.1,       # Overlap between processing windows
                 buffer_size: int = 30):            # Number of past frames to keep for context
        self.model_name = model_name
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.overlap_factor = overlap_factor
        self.buffer_size = buffer_size

class VADResult:
    """Result from VAD processing"""
    def __init__(self, is_speech: bool = False, confidence: float = 0.0, timestamps: Optional[List[Dict[str, float]]] = None):
        self.is_speech = is_speech
        self.confidence = confidence
        self.timestamps = timestamps or []

def load_silero_vad():
    """Load the Silero VAD model from torch hub"""
    
    # Check if model is already cached
    if _vad_cache["model"] is not None:
        return _vad_cache["model"]
    
    try:
        print("Loading Silero VAD model...")
        start_time = time.time()
        
        # Load model and utils directly from torch hub
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        # Extract the get_speech_timestamps function from utils
        _vad_cache["speech_timestamps_fn"] = utils[0]
        
        # Cache model for reuse
        _vad_cache["model"] = model
        
        load_time = time.time() - start_time
        print(f"Silero VAD model loaded in {load_time:.2f}s")
        
        return model
    except Exception as e:
        print(f"Error loading Silero VAD model: {e}")
        raise

def get_speech_timestamps(audio_tensor, model, sampling_rate=16000, 
                          threshold=0.3, min_speech_duration_ms=250, 
                          min_silence_duration_ms=100, speech_pad_ms=30,
                          return_seconds=False):
    """
    Get speech timestamps using the Silero VAD model
    
    Args:
        audio_tensor: Audio tensor (1D)
        model: Silero VAD model
        sampling_rate: Audio sampling rate
        threshold: Speech threshold (higher = stricter)
        min_speech_duration_ms: Minimum speech segment duration (ms)
        min_silence_duration_ms: Minimum silence duration between segments (ms)
        speech_pad_ms: Extra padding for each speech segment (ms)
        return_seconds: Whether to return timestamps in seconds (default is frames)
        
    Returns:
        List of dicts with 'start' and 'end' timestamps
    """
    
    # Ensure we have the get_speech_timestamps function
    if _vad_cache["speech_timestamps_fn"] is None:
        # If we have the model but not the function, reload the model to get the utils
        if _vad_cache["model"] is not None:
            _, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            _vad_cache["speech_timestamps_fn"] = utils[0]
        else:
            # If we don't have the model, we need to load it first
            load_silero_vad()
    
    # Get speech timestamps function from cache
    get_speech_ts_fn = _vad_cache["speech_timestamps_fn"]
    
    # Ensure the tensor is 1D
    if len(audio_tensor.shape) > 1:
        audio_tensor = audio_tensor.squeeze()
    
    # Convert to float32 if not already
    if audio_tensor.dtype != torch.float32:
        audio_tensor = audio_tensor.float()
    
    # Ensure normalized between -1 and 1
    max_val = torch.max(torch.abs(audio_tensor))
    if max_val > 1.0:
        audio_tensor = audio_tensor / max_val
    
    # Apply a small smoothing to prevent noise spikes
    if audio_tensor.shape[0] >= 32:
        # Simple moving average for very short tensors
        kernel_size = min(5, audio_tensor.shape[0] // 8)
        if kernel_size > 0:
            padded = torch.nn.functional.pad(audio_tensor.view(1, 1, -1), (kernel_size//2, kernel_size//2), mode='replicate')
            smoothed = torch.nn.functional.avg_pool1d(padded, kernel_size, stride=1).view(-1)
            # Blend original with smoothed to preserve details
            audio_tensor = 0.7 * audio_tensor + 0.3 * smoothed
    
    # Get speech timestamps using the utility function
    try:
        timestamps = get_speech_ts_fn(
            audio_tensor, 
            model,
            sampling_rate=sampling_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=return_seconds
        )
        return timestamps
    except Exception as e:
        print(f"Error getting speech timestamps: {e}")
        return []

class VADHandler:
    """Handles voice activity detection in audio streams"""
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self.vad_model = None
        self.initialized = False
        self.buffer = []
        self.last_is_speech = False
        self.last_confidence = 0.0
        self.silence_start = None
        self.speech_start = None
        
        # Add a buffer for continuous audio processing
        self.audio_buffer = np.array([], dtype=np.int16)
        self.overlap_samples = int(self.config.window_size_samples * self.config.overlap_factor)
        
        # Initialize eagerly
        self.initialize()
        
    def initialize(self) -> bool:
        """Initialize the VAD model"""
        if self.initialized:
            return True
            
        try:
            print("Loading VAD model...")
            # Use our inlined load_silero_vad function
            self.vad_model = load_silero_vad()
            self.initialized = True
            print("VAD model loaded successfully")
            return True
        except Exception as e:
            print(f"Error initializing VAD model: {e}")
            return False
    
    def process_chunk(self, audio_chunk: bytes) -> VADResult:
        """Process an audio chunk and detect speech"""
        if not self.initialized:
            if not self.initialize():
                return VADResult()
        
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Check for incorrect audio length/sample rate issues
            # If the audio chunk has unusually many samples for a small chunk, 
            # it might be recorded at a higher rate than we expect
            expected_samples = self.config.window_size_samples  # Expect roughly this many samples
            actual_samples = len(audio_np)
            
            # If the chunk is significantly larger than expected, we may need to downsample
            if actual_samples > 2.5 * expected_samples:
                print(f"WARNING: Large audio chunk detected ({actual_samples} samples, expected ~{expected_samples})")
                print("This might indicate a sample rate mismatch. Attempting to correct...")
                
                # Guess the source sample rate based on the ratio
                # Common rates are 44100, 48000, etc.
                likely_source_rate = None
                for rate in [44100, 48000, 96000, 22050]:
                    ratio = rate / self.config.sample_rate
                    if 0.8 < actual_samples / (expected_samples * ratio) < 1.2:
                        likely_source_rate = rate
                        break
                
                if likely_source_rate:
                    print(f"Audio appears to be at {likely_source_rate}Hz instead of {self.config.sample_rate}Hz")
                    
                    # Resample to the correct rate
                    import scipy.signal as signal
                    target_samples = int(actual_samples * self.config.sample_rate / likely_source_rate)
                    resampled = signal.resample(audio_np, target_samples)
                    audio_np = np.int16(resampled)
                    audio_chunk = audio_np.tobytes()
                    print(f"Resampled from {actual_samples} to {len(audio_np)} samples")
                    
                    # Save a debug copy of the resampled audio
                    try:
                        import wave
                        import os
                        from datetime import datetime
                        
                        # Create debug directory
                        debug_dir = os.path.join(os.getcwd(), "debug_audio")
                        os.makedirs(debug_dir, exist_ok=True)
                        
                        # Generate filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(debug_dir, f"vad_resampled_{timestamp}.wav")
                        
                        with wave.open(filename, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(self.config.sample_rate)
                            wf.writeframes(audio_chunk)
                        print(f"Saved resampled audio to {filename}")
                    except Exception as e:
                        print(f"Error saving resampled audio: {e}")
            
            # Add to buffer for continuous processing
            self.audio_buffer = np.append(self.audio_buffer, audio_np)
            
            # Keep buffer at a reasonable size, but maintain overlapping context
            max_buffer_samples = self.config.buffer_size * self.config.window_size_samples
            if len(self.audio_buffer) > max_buffer_samples:
                # Keep a portion of the buffer for context
                ctx_samples = min(max_buffer_samples // 2, 4096)
                self.audio_buffer = self.audio_buffer[-max_buffer_samples:]
            
            # Convert to float and normalize
            audio_float = self.audio_buffer.astype(np.float32) / 32768.0
            
            # Convert to PyTorch tensor
            audio_tensor = torch.tensor(audio_float)
            
            # Check if we have enough data for processing
            if len(audio_tensor) < self.config.window_size_samples:
                # Not enough data yet
                return VADResult(is_speech=self.last_is_speech, confidence=self.last_confidence)
            
            # Use our inlined get_speech_timestamps function
            timestamps = get_speech_timestamps(
                audio_tensor, 
                self.vad_model,
                sampling_rate=self.config.sample_rate,
                threshold=self.config.threshold,
                min_speech_duration_ms=int(self.config.min_speech_duration * 1000),
                min_silence_duration_ms=int(self.config.min_silence_duration * 1000),
                speech_pad_ms=self.config.speech_pad_ms,
                return_seconds=True
            )
            
            # Determine if speech is present based on timestamps
            is_speech = len(timestamps) > 0
            
            # Calculate confidence
            confidence = 0.5  # Default confidence
            
            # If speech detected, use the latest timestamp's end time
            # to determine if it's current
            if is_speech:
                # Get the latest timestamp
                latest = timestamps[-1]
                
                # Check if the speech is current (end is close to the end of the buffer)
                end_sample = int(latest["end"] * self.config.sample_rate)
                buffer_end = len(self.audio_buffer)
                
                # If end is within 0.5 seconds of buffer end, it's current speech
                if buffer_end - end_sample < 0.5 * self.config.sample_rate:
                    confidence = 0.95
                else:
                    # It's old speech, not current
                    is_speech = False
                    confidence = 0.1
                
                # Update speech start if needed
                if not self.speech_start and is_speech:
                    self.speech_start = time.time()
                    self.silence_start = None
            else:
                # If no speech detected
                if self.last_is_speech:
                    # Just transitioned to silence
                    self.silence_start = time.time()
                elif self.silence_start:
                    # Check if silence has been long enough to confirm no speech
                    silence_duration = time.time() - self.silence_start
                    if silence_duration < self.config.min_silence_duration:
                        # Not silent long enough to be sure, maintain speech state
                        # This helps prevent choppy detections
                        is_speech = self.last_is_speech
                        confidence = max(0.6 - (silence_duration / self.config.min_silence_duration * 0.4), 0.2)
                
                # Reset speech start if we're sure it's silent
                if not is_speech and self.silence_start and time.time() - self.silence_start > self.config.min_silence_duration:
                    self.speech_start = None
            
            # Store state for next call
            self.last_is_speech = is_speech
            self.last_confidence = confidence
            
            # Return result
            return VADResult(
                is_speech=is_speech,
                confidence=confidence,
                timestamps=[{
                    "start": ts["start"], 
                    "end": ts["end"]
                } for ts in timestamps]
            )
            
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            import traceback
            traceback.print_exc()
            return VADResult(is_speech=self.last_is_speech, confidence=self.last_confidence)
            
    def reset(self):
        """Reset the VAD state"""
        self.buffer = []
        self.audio_buffer = np.array([], dtype=np.int16)
        self.last_is_speech = False
        self.last_confidence = 0.0
        self.silence_start = None
        self.speech_start = None 