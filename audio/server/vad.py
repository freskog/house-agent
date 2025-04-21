"""
Voice Activity Detection (VAD) module for detecting speech in audio.
"""

import os
import torch
import numpy as np
import time
from typing import Optional, Dict, List, Tuple, Any, Union
import sys
import collections
from langsmith import traceable

# Cache for models
_vad_cache = {
    "model": None,
    "speech_timestamps_fn": None
}

class VADConfig:
    """Configuration for VAD"""
    def __init__(self, 
                 model_name: str = "silero_vad",
                 threshold: float = 0.5,  # Dramatically increased from 0.2 to 0.5 to be much less sensitive
                 sample_rate: int = 16000,
                 min_speech_duration: float = 0.8,  # Doubled from 0.4 to require longer speech segments
                 min_silence_duration: float = 1.0,  # Increased from 0.7 to require more definitive silence
                 window_size_samples: int = 1536,
                 speech_pad_ms: int = 80,           # Further reduced from 120 to minimize padding
                 overlap_factor: float = 0.1,
                 buffer_size: int = 8,              # Reduced from 10 to accumulate less potential noise
                 verbose: bool = False):            # New parameter to control logging
        self.model_name = model_name
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.overlap_factor = overlap_factor
        self.buffer_size = buffer_size
        self.verbose = verbose

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

@traceable(run_type="chain", name="VAD_Get_Speech_Timestamps")
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
        self.last_is_speech = False
        self.last_confidence = 0.0
        self.silence_start = time.time()  # Initialize with current time to avoid NoneType error
        self.speech_start = time.time()  # Initialize with current time
        
        # Add a buffer for continuous audio processing
        self.audio_buffer = np.array([], dtype=np.int16)
        
        # Add a chunk buffer for aggregating multiple audio chunks for better speech detection
        self.chunk_buffer = collections.deque(maxlen=self.config.buffer_size)
        
        # Frame counter for logging
        self._frame_counter = 0
        
        # Initialize eagerly
        self.initialize()
        
    def initialize(self) -> bool:
        """Initialize the VAD model"""
        if self.initialized:
            return True
            
        try:
            if self.config.verbose:
                print("Loading VAD model...")
            # Use our inlined load_silero_vad function
            self.vad_model = load_silero_vad()
            self.initialized = True
            if self.config.verbose:
                print("VAD model loaded successfully")
            return True
        except Exception as e:
            print(f"Error initializing VAD model: {e}")
            return False
    
    def process_chunk(self, audio_chunk: bytes) -> VADResult:
        """Process an audio chunk and detect speech"""
        if not self.initialized:
            if not self.initialize():
                print("WARNING: VAD not initialized in process_chunk")
                return VADResult()
        
        try:
            # Increment frame counter
            self._frame_counter += 1
            
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Calculate RMS value to check if audio is too quiet to process
            if len(audio_np) > 0:
                audio_rms = np.sqrt(np.mean(np.power(audio_np.astype(np.float32), 2)))
                
                # Skip very quiet frames entirely (RMS < 200 is quite quiet)
                if audio_rms < 200:
                    return VADResult(is_speech=False, confidence=0.0)
            
            # Add to chunk buffer for better context
            if len(audio_np) > 0:
                self.chunk_buffer.append(audio_np)
            else:
                print("WARNING: Empty audio chunk received")
                return VADResult(is_speech=self.last_is_speech, confidence=self.last_confidence * 0.9)
            
            # Check for incorrect audio length/sample rate issues
            expected_samples = self.config.window_size_samples
            actual_samples = len(audio_np)
            
            # If the chunk is significantly larger than expected, we may need to downsample
            if actual_samples > 2.5 * expected_samples:
                # Existing resampling code preserved here
                if self.config.verbose:
                    print(f"WARNING: Large audio chunk detected ({actual_samples} samples, expected ~{expected_samples})")
                    print("This might indicate a sample rate mismatch. Attempting to correct...")
                
                likely_source_rate = None
                for rate in [44100, 48000, 96000, 22050]:
                    ratio = rate / self.config.sample_rate
                    if 0.8 < actual_samples / (expected_samples * ratio) < 1.2:
                        likely_source_rate = rate
                        break
                
                if likely_source_rate:
                    if self.config.verbose:
                        print(f"Audio appears to be at {likely_source_rate}Hz instead of {self.config.sample_rate}Hz")
                    
                    # Resample to the correct rate
                    import scipy.signal as signal
                    target_samples = int(actual_samples * self.config.sample_rate / likely_source_rate)
                    resampled = signal.resample(audio_np, target_samples)
                    audio_np = np.int16(resampled)
                    audio_chunk = audio_np.tobytes()
                    if self.config.verbose:
                        print(f"Resampled from {actual_samples} to {len(audio_np)} samples")
                    
                    # Update the buffer with the resampled audio
                    self.chunk_buffer[-1] = audio_np
            
            # Process multiple chunks together for better context
            # Only if we have enough chunks or the audio is loud
            audio_mean = np.mean(np.abs(audio_np))
            is_loud = audio_mean > 500  # Higher threshold for what's considered a loud frame
            
            # Default values in case processing fails
            is_speech = False
            confidence = 0.0
            timestamps = []
            
            if len(self.chunk_buffer) >= 3 or is_loud:
                # Combine chunks for more context
                combined_audio = np.concatenate(list(self.chunk_buffer))
                audio_tensor = torch.tensor(combined_audio)
                
                # Process through VAD
                try:
                    # For better accuracy, we use get_speech_timestamps when possible
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
                    
                    # Calculate confidence as proportion of audio marked as speech
                    if len(timestamps) > 0:
                        total_duration = len(combined_audio) / self.config.sample_rate
                        speech_duration = sum((ts['end'] - ts['start']) for ts in timestamps)
                        confidence = min(1.0, speech_duration / total_duration if total_duration > 0 else 0)
                        
                        # More aggressive confidence threshold
                        is_speech = confidence > 0.4  # Require at least 40% of the audio to be speech
                        
                        if is_speech:
                            confidence = max(0.7, confidence)  # Higher minimum confidence when speech is detected
                            
                            # Only log when speech is detected if verbose is enabled
                            if self.config.verbose:
                                print(f"Debug: SPEECH DETECTED! Confidence: {confidence:.2f}, Timestamps: {timestamps}")
                    
                    # Update state with results - include hysteresis logic only here
                    if is_speech:
                        self.last_is_speech = True
                        self.last_confidence = confidence
                        self.speech_start = time.time()
                    else:
                        # Only transition to not-speech after a delay (hysteresis)
                        if self.last_is_speech and (time.time() - self.speech_start) < self.config.min_speech_duration:
                            # Too soon after speech started, don't transition yet
                            is_speech = True
                            confidence = self.last_confidence * 0.8  # Decay confidence
                        else:
                            self.last_is_speech = False
                            self.last_confidence = 0.0
                            self.silence_start = time.time()
                    
                    # Return result
                    result = VADResult(
                        is_speech=is_speech,
                        confidence=confidence,
                        timestamps=timestamps
                    )
                    
                    return result
                    
                except Exception as e:
                    print(f"Error in VAD processing: {e}")
                    if self.config.verbose:
                        import traceback
                        traceback.print_exc()
                    # Use fallback values which were initialized earlier
            
            # If we don't have enough chunks yet or processing failed, return results based on state
            return VADResult(is_speech=self.last_is_speech, confidence=self.last_confidence * 0.9)
            
        except Exception as e:
            print(f"Error in VAD handler: {e}")
            import traceback
            traceback.print_exc()
            return VADResult()
    
    def reset(self):
        """Reset the VAD state"""
        self.chunk_buffer.clear()
        self.last_is_speech = False
        self.last_confidence = 0.0
        self.silence_start = time.time()
        self.speech_start = time.time()
        self.audio_buffer = np.array([], dtype=np.int16) 