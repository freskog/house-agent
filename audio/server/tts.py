"""
Text-to-Speech module for generating audio from text.
"""

import asyncio
import io
import soundfile as sf
import numpy as np
import functools
import time
from typing import Optional, List, Dict, Any, Tuple
import torch

class TTSConfig:
    """Configuration for TTS"""
    def __init__(self, 
                 voice: str = 'af_heart',
                 lang_code: str = 'a',
                 sample_rate: int = 24000,
                 cache_size: int = 100):
        self.voice = voice
        self.lang_code = lang_code  # 'a' for American English
        self.sample_rate = sample_rate
        self.cache_size = cache_size  # Number of items to cache

class TTSEngine:
    """Handles text-to-speech conversion"""
    
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self.tts_pipeline = None
        self.initialized = False
        # Cache for generated audio
        self.cache = {}
        self.cache_keys = []  # To track insertion order
        # Initialize eagerly
        self.initialize()
        
    def initialize(self) -> bool:
        """Initialize the TTS engine"""
        if self.initialized:
            return True
            
        try:
            # Import here to avoid dependency if not used
            from kokoro import KPipeline
            
            print(f"Loading Kokoro TTS engine with language code '{self.config.lang_code}'...")
            self.tts_pipeline = KPipeline(lang_code=self.config.lang_code)
            print("Kokoro TTS engine loaded successfully")
            self.initialized = True
            return True
        except ImportError as e:
            print(f"Kokoro TTS not available: {e}")
            return False
        except Exception as e:
            print(f"Error initializing TTS engine: {e}")
            return False
    
    async def synthesize(self, text: str, voice: Optional[str] = None) -> bytes:
        """Convert text to speech audio bytes"""
        if not self.initialized:
            print("TTS engine not initialized, cannot synthesize speech")
            return b''
                
        # Use specified voice or fall back to config
        voice_to_use = voice or self.config.voice
        
        # Create cache key
        cache_key = f"{text}:{voice_to_use}"
        
        # Check cache first
        if cache_key in self.cache:
            print(f"TTS cache hit for text: '{text[:70]}...' (if longer)")
            return self.cache[cache_key]
            
        try:
            start_time = time.time()
            
            # Run generation in a thread to avoid blocking
            audio_data = await asyncio.to_thread(
                self._generate_audio, 
                text, 
                voice_to_use
            )
            
            process_time = time.time() - start_time
            data_size = len(audio_data)
            
            if data_size < 100:
                print(f"WARNING: TTS generated empty or very small audio ({data_size} bytes) for: '{text[:30]}...'")
            else:
                print(f"TTS generated {data_size} bytes in {process_time:.2f}s for: '{text[:30]}...' (if longer)")
            
            # Cache the result
            self._cache_result(cache_key, audio_data)
            
            return audio_data
            
        except Exception as e:
            print(f"Error in TTS synthesis: {e}")
            return b''
            
    def _cache_result(self, key: str, audio_data: bytes):
        """Add result to cache, managing cache size"""
        # Add to cache
        self.cache[key] = audio_data
        self.cache_keys.append(key)
        
        # Trim cache if needed
        if len(self.cache_keys) > self.config.cache_size:
            oldest_key = self.cache_keys.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
            
    def _generate_audio(self, text: str, voice: str) -> bytes:
        """Generate audio from text using Kokoro (runs in thread)"""
        try:
            # Generate audio
            generator = self.tts_pipeline(text, voice=voice)
            audio = None
            
            # Process generator output (only take the first chunk for now)
            # In the future, we could combine multiple chunks for longer text
            for i, (gs, ps, audio_chunk) in enumerate(generator):
                if i == 0:  # Just take the first chunk for now
                    audio = audio_chunk
                    break
                    
            # If no audio was generated, return empty bytes
            if audio is None:
                print("No audio was generated by Kokoro TTS")
                return b''
                
            # Convert PyTorch tensor to NumPy if it's a tensor
            if isinstance(audio, torch.Tensor):
                print(f"Converting torch.Tensor of shape {audio.shape} to NumPy array")
                audio = audio.detach().cpu().numpy()
            
            # Check if audio has non-zero values
            audio_max = np.abs(audio).max()
            audio_min = np.abs(audio).min()
            audio_mean = np.mean(np.abs(audio))
            print(f"TTS audio stats: shape={audio.shape}, dtype={audio.dtype}, min={audio_min:.6f}, max={audio_max:.6f}, mean={audio_mean:.6f}")
            
            if audio_max < 0.01:
                print(f"Warning: Audio signal is very weak (max amplitude: {audio_max:.4f})")
            
            # Normalize audio to prevent distortion
            if audio_max > 1.0:
                print(f"Normalizing audio (max value: {audio_max:.4f})")
                audio = audio / audio_max
                
            # Convert from float to int16
            audio_int16 = np.int16(audio * 32767)
                
            # Create WAV file directly with strict control over format
            with io.BytesIO() as wav_io:
                # Write WAV header and data using the wave module
                import wave
                with wave.open(wav_io, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.config.sample_rate)  # Use config sample rate
                    wf.writeframes(audio_int16.tobytes())
                    
                # Get the WAV data
                wav_data = wav_io.getvalue()
                
                # Verify the WAV file by reading it back
                try:
                    with io.BytesIO(wav_data) as verify_io:
                        with wave.open(verify_io, 'rb') as wf:
                            channels = wf.getnchannels()
                            sample_width = wf.getsampwidth()
                            frame_rate = wf.getframerate()
                            n_frames = wf.getnframes()
                            
                            # Check if the WAV file matches our expectations
                            if frame_rate != self.config.sample_rate:
                                print(f"WARNING: Generated WAV has sample rate {frame_rate}Hz but expected {self.config.sample_rate}Hz")
                            
                            print(f"Generated WAV: {channels} channels, {frame_rate}Hz, {sample_width} bytes/sample, {n_frames} frames")
                except Exception as e:
                    print(f"Error verifying WAV data: {e}")
                
                # Print WAV header information for debugging
                wav_header = wav_data[:44]  # WAV header is 44 bytes
                header_hex = ' '.join([f'{b:02x}' for b in wav_header[:20]])  # First 20 bytes is enough for key info
                print(f"WAV header (first 20 bytes): {header_hex}")
                print(f"WAV size: {len(wav_data)} bytes")
                
                return wav_data
                
        except Exception as e:
            print(f"Error in TTS generation: {e}")
            import traceback
            traceback.print_exc()
            return b''

    def get_available_voices(self) -> List[str]:
        """Get list of available voices"""
        # Currently available Kokoro voices (as of v0.9.4)
        return [
            'af_heart',    # Female American English
            'am_bear',     # Male American English
            'bf_heart',    # Female British English
            'bm_bear',     # Male British English
            'jf_heart',    # Female Japanese
            'jm_bear',     # Male Japanese
            'if_heart',    # Female Italian
            'im_bear',     # Male Italian
            'ff_heart',    # Female French
            'fm_bear',     # Male French
            'sf_heart',    # Female Spanish
            'sm_bear',     # Male Spanish
            'cf_heart',    # Female Chinese
            'cm_bear',     # Male Chinese
            'hf_heart',    # Female Hindi
            'hm_bear',     # Male Hindi
            'pf_heart',    # Female Portuguese
            'pm_bear'      # Male Portuguese
        ] 