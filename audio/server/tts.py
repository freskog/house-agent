"""
Text-to-Speech module for generating audio from text using Kokoro.
"""

import asyncio
import io
import soundfile as sf
import numpy as np
import functools
import time
from typing import Optional, List, Dict, Any, Tuple
import re
import torch

# Import Kokoro TTS
from kokoro import KPipeline

class TTSConfig:
    """Configuration for text-to-speech"""
    def __init__(self):
        self.voice = "af_heart"    # Default voice (American female heart)
        self.lang_code = "a"       # Default language (American English)
        self.sample_rate = 24000   # Sample rate in Hz
        self.cache_size = 100      # Max cache entries
        self.speed = 1.0           # Speech speed

class TTSEngine:
    """Handles text-to-speech conversion using Kokoro"""
    
    def __init__(
        self, 
        voice="af_heart",
        lang_code="a",  # 'a' for American English
        speed=1.0,
        config: Optional[TTSConfig] = None
    ):
        """Initialize the TTS engine
        
        Args:
            voice: Voice to use (e.g., "af_heart")
            lang_code: Language code ('a' for American English)
            speed: Speech speed
            config: Optional TTSConfig object
        """
        # Process the config if provided
        if config:
            self.voice = config.voice if hasattr(config, 'voice') else voice
            self.lang_code = config.lang_code if hasattr(config, 'lang_code') else lang_code
            self.speed = config.speed if hasattr(config, 'speed') else speed
        else:
            self.voice = voice
            self.lang_code = lang_code
            self.speed = speed
            
        self.pipeline = None
        self.initialized = False  # Track initialization status
        
        # Initialize the model
        self._setup_tts()
        
    def _setup_tts(self):
        """Set up the Kokoro TTS pipeline"""
        print(f"Initializing Kokoro TTS engine with voice {self.voice} and language {self.lang_code}")
        try:
            # Create the Kokoro pipeline
            self.pipeline = KPipeline(lang_code=self.lang_code)
            print("Kokoro pipeline created successfully")
            self.initialized = True
            
        except Exception as e:
            print(f"Error initializing Kokoro TTS: {e}")
            self.pipeline = None
            self.initialized = False
            
    def _preprocess_text(self, text):
        """Preprocess text for TTS - sentence splitting, normalization etc."""
        try:
            # First protect common abbreviations from being split
            protected_text = text
            abbreviations = ["Mr.", "Mrs.", "Dr.", "Ph.D.", "M.D.", "B.A.", "B.S.", "M.S.", "i.e.", "e.g.", "U.S.A.", "U.K.", "St."]
            
            for abbr in abbreviations:
                protected_text = protected_text.replace(abbr, abbr.replace(".", "{{DOT}}"))
            
            # Now split on sentence boundaries
            split_pattern = r'(?<=[.!?])\s+'
            sentences = re.split(split_pattern, protected_text)
            
            # Restore the protected abbreviations
            sentences = [s.replace("{{DOT}}", ".") for s in sentences]
            
            # Handle edge cases: very long sentences may need further splitting
            max_sentence_length = 200  # Characters
            final_sentences = []
            
            for s in sentences:
                if len(s) > max_sentence_length:
                    # Split long sentences at commas or other natural breaks
                    comma_splits = re.split(r'(?<=,|\;)\s+', s)
                    if len(comma_splits) > 1 and max(len(cs) for cs in comma_splits) < max_sentence_length:
                        final_sentences.extend(comma_splits)
                    else:
                        # Still too long, use a simple length-based split as last resort
                        final_sentences.extend([s[i:i+max_sentence_length] for i in range(0, len(s), max_sentence_length)])
                else:
                    final_sentences.append(s)
            
            print(f"Split text into {len(final_sentences)} sentences")
            return final_sentences if final_sentences else [text]
        
        except Exception as e:
            print(f"Error during text preprocessing: {e}")
            # Safest fallback is to return the whole text as one chunk
            return [text]
    
    async def _generate_audio(self, text):
        """Generate audio from text using Kokoro pipeline"""
        if not self.pipeline:
            print("Kokoro pipeline not initialized")
            return None
            
        try:
            # Use event loop to run CPU-bound TTS in a thread pool
            loop = asyncio.get_event_loop()
            
            # Define the function to run in the thread pool
            def process_text():
                all_audio = []
                # Create a generator from the pipeline
                generator = self.pipeline(text, voice=self.voice, speed=self.speed)
                
                # Process all outputs from the generator
                for i, (gs, ps, audio) in enumerate(generator):
                    all_audio.append(audio)
                
                # Combine all audio segments if needed
                if len(all_audio) > 1:
                    return np.concatenate(all_audio)
                elif len(all_audio) == 1:
                    return all_audio[0]
                else:
                    return None
            
            # Run in thread pool
            audio_data = await loop.run_in_executor(None, process_text)
            
            if audio_data is None:
                print("No audio generated")
                return None
                
            # Convert to bytes (WAV format)
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, 24000, format="WAV", subtype="PCM_16")
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            print(f"Kokoro TTS error: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    async def synthesize(self, text):
        """Convert text to speech and return audio data
        
        Args:
            text: Text to convert to speech
            
        Returns:
            bytes: Audio data (WAV format)
        """
        if not text:
            print("Empty text provided to synthesize method")
            return None
            
        # Split text into chunks if needed
        chunks = self._preprocess_text(text)
        
        if not chunks:
            print("Text preprocessing returned no chunks")
            return None
            
        print(f"Processing {len(chunks)} text chunks for TTS")
        
        audio_chunks = []
        total_len = 0
        
        # Process each chunk in sequence
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                print(f"Skipping empty chunk {i+1}/{len(chunks)}")
                continue
                
            print(f"Generating audio for chunk {i+1}/{len(chunks)}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")
            audio_data = await self._generate_audio(chunk)
            
            if audio_data:
                audio_chunks.append(audio_data)
                total_len += len(audio_data)
                print(f"Generated {len(audio_data)} bytes of audio for chunk {i+1}")
            else:
                print(f"Failed to generate audio for chunk {i+1}")
        
        if not audio_chunks:
            print("No audio chunks were generated")
            return None
        
        # If only one chunk, return it directly
        if len(audio_chunks) == 1:
            print("Returning single audio chunk")
            return audio_chunks[0]
        
        # Otherwise, concatenate all chunks
        print(f"Combining {len(audio_chunks)} audio chunks (total {total_len} bytes)")
        combined_audio = self._combine_audio_chunks(audio_chunks)
        
        if combined_audio:
            print(f"Combined audio size: {len(combined_audio)} bytes")
        else:
            print("Failed to combine audio chunks")
            # Fallback to the first chunk if combination fails
            if audio_chunks:
                print("Falling back to first audio chunk")
                return audio_chunks[0]
        
        return combined_audio
    
    def _combine_audio_chunks(self, audio_chunks):
        """Combine multiple WAV audio chunks into a single audio stream
        
        Args:
            audio_chunks: List of WAV audio data bytes
            
        Returns:
            bytes: Combined audio data
        """
        if not audio_chunks:
            print("No audio chunks to combine")
            return None
            
        # If only one chunk, return it directly
        if len(audio_chunks) == 1:
            return audio_chunks[0]
            
        try:
            # Read all wav data
            wav_data = []
            sample_rate = None
            
            for i, chunk in enumerate(audio_chunks):
                try:
                    # Read the audio data from the buffer
                    buffer = io.BytesIO(chunk)
                    data, rate = sf.read(buffer)
                    
                    # Store the first sample rate we encounter
                    if sample_rate is None:
                        sample_rate = rate
                        
                    # Ensure all chunks have the same sample rate
                    if rate != sample_rate:
                        print(f"Warning: Sample rate mismatch in chunk {i+1} ({rate} vs {sample_rate})")
                        # Could resample here if needed in the future
                        
                    wav_data.append(data)
                except Exception as e:
                    print(f"Error processing audio chunk {i+1}: {e}")
                    # Skip problematic chunks but continue with the rest
                    continue
            
            if not wav_data:
                print("No valid audio data found in chunks")
                return None
                
            # Concatenate all audio data
            print(f"Concatenating {len(wav_data)} audio arrays")
            combined = np.concatenate(wav_data)
            
            # Convert back to wav bytes
            buffer = io.BytesIO()
            sf.write(buffer, combined, sample_rate, format="WAV", subtype="PCM_16")
            buffer.seek(0)
            
            result = buffer.read()
            print(f"Successfully combined audio: {len(result)} bytes")
            return result
            
        except Exception as e:
            print(f"Error combining audio chunks: {e}")
            # In case of error, return the first chunk as fallback
            if audio_chunks:
                print("Error during audio combining, falling back to first chunk")
                return audio_chunks[0]
            return None

    def get_available_voices(self) -> List[str]:
        """Get list of available voices"""
        # These are the most common Kokoro voices
        voices = [
            # American English Female
            "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", 
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            # American English Male
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
            "am_michael", "am_onyx", "am_puck",
            # British English
            "bf_alice", "bf_emma", "bf_isabella", "bf_lily", 
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis"
        ]
        return voices

    def is_initialized(self) -> bool:
        """Check if the TTS engine is initialized"""
        return self.initialized and self.pipeline is not None 