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
from langsmith import traceable

# Import our logging infrastructure
from utils.logging_config import setup_logging

# Set up logger for this module
logger = setup_logging(__name__)

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
        logger.info(f"🔊 Initializing Kokoro TTS engine with voice {self.voice} and language {self.lang_code}")
        try:
            # Create the Kokoro pipeline
            self.pipeline = KPipeline(lang_code=self.lang_code)
            logger.info("✅ Kokoro pipeline created successfully")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"❌ Error initializing Kokoro TTS: {e}")
            self.pipeline = None
            self.initialized = False
            
    def _preprocess_text(self, text):
        """Preprocess text for TTS - sentence splitting, normalization etc.
        
        This method splits text into natural speech units (sentences) based only on
        periods, question marks, and exclamation marks. It preserves natural flow
        and conversational tone by keeping phrases within sentences together.
        
        Note: When generating prompts for this TTS system, instruct the LLM to:
        1. Phrase responses conversationally, as a human speaking to another human
        2. Avoid listing items with numbers or bullet points
        3. Use complete, flowing sentences rather than fragmented responses
        """
        try:
            # First protect common abbreviations from being split incorrectly
            protected_text = text
            abbreviations = ["Mr.", "Mrs.", "Dr.", "Ph.D.", "M.D.", "B.A.", "B.S.", "M.S.", "i.e.", "e.g.", "U.S.A.", "U.K.", "St."]
            
            for abbr in abbreviations:
                protected_text = protected_text.replace(abbr, abbr.replace(".", "{{DOT}}"))
            
            # Split ONLY on sentence boundaries (periods, exclamation marks, question marks)
            sentence_boundaries = r'(?<=[.!?])\s+'
            sentences = re.split(sentence_boundaries, protected_text)
            
            # Process each sentence
            final_segments = []
            
            for sentence in sentences:
                # Restore the protected abbreviations
                sentence = sentence.replace("{{DOT}}", ".")
                
                # Add the sentence as is without any length-based splitting
                if sentence.strip():
                    final_segments.append(sentence.strip())
            
            # For very long texts, still prioritize the first few segments
            # This makes the system respond faster with the beginning of the content
            if len(final_segments) > 10:
                # Rearrange segments to prioritize the first few
                prioritized_segments = final_segments[:3]
                prioritized_segments.extend(final_segments[3::2])  # Every other segment after the first 3
                prioritized_segments.extend(final_segments[4::2])  # The remaining segments
                final_segments = prioritized_segments
            
            logger.debug(f"📝 Split text into {len(final_segments)} segments for natural speech flow")
            return final_segments if final_segments else [text]
        
        except Exception as e:
            logger.error(f"❌ Error during text preprocessing: {e}")
            import traceback
            traceback.print_exc()
            # Safest fallback is to return the whole text as one chunk
            return [text]
    
    @traceable(run_type="chain", name="TTS_Generate_Audio_Chunk")
    async def _generate_audio_numpy(self, text):
        """Generate audio from text using Kokoro pipeline, returning numpy array
        
        Optimized for single sentences or small chunks of text.
        Returns raw numpy array to avoid WAV conversion overhead.
        """
        try:
            if not self.pipeline:
                logger.error("❌ Kokoro pipeline not initialized")
                return None
                
            # Use event loop to run CPU-bound TTS in a thread pool
            loop = asyncio.get_event_loop()
            
            # Define the function to run in the thread pool - optimized for single sentence
            def process_text():
                try:
                    # Create a generator from the pipeline
                    generator = self.pipeline(text, voice=self.voice, speed=self.speed)
                    
                    # Collect audio segments for this sentence
                    all_audio = []
                    
                    # Process all outputs from the generator
                    for _, _, audio in generator:
                        all_audio.append(audio)
                    
                    # Combine audio segments
                    if len(all_audio) > 1:
                        return np.concatenate(all_audio)
                    elif len(all_audio) == 1:
                        return all_audio[0]
                    else:
                        return None
                except Exception as e:
                    logger.error(f"❌ Error in TTS pipeline: {e}")
                    return None
            
            # Run in thread pool with a timeout to prevent hanging
            try:
                audio_data = await asyncio.wait_for(
                    loop.run_in_executor(None, process_text),
                    timeout=10.0  # 10 second timeout per sentence
                )
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Timeout generating audio for: '{text[:30]}...'")
                return None
            
            if audio_data is None:
                logger.warning("⚠️ No audio generated")
                return None
                
            # Return raw numpy array - no WAV conversion here!
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Kokoro TTS error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _generate_audio(self, text):
        """Legacy method that converts numpy to WAV - kept for backward compatibility"""
        audio_np = await self._generate_audio_numpy(text)
        if audio_np is None:
            return None
            
        # Convert to bytes (WAV format) only when needed
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, 24000, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        
        return buffer.read()
    
    def _combine_audio_numpy(self, audio_arrays):
        """Combine multiple numpy audio arrays efficiently
        
        Args:
            audio_arrays: List of numpy audio arrays
            
        Returns:
            numpy.ndarray: Combined audio array
        """
        if not audio_arrays:
            logger.error("No audio arrays to combine")
            return None
            
        # If only one array, return it directly
        if len(audio_arrays) == 1:
            return audio_arrays[0]
            
        try:
            # Simple numpy concatenation - much more efficient than WAV parsing
            logger.debug(f"Concatenating {len(audio_arrays)} audio arrays...")
            combined = np.concatenate(audio_arrays)
            logger.debug(f"Combined audio: {len(combined)} samples")
            return combined
            
        except Exception as e:
            logger.error(f"Error combining audio arrays: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    @traceable(run_type="chain", name="TTS_Synthesize_Speech")
    async def synthesize(self, text):
        """Convert text to speech and return audio data
        
        Args:
            text: Text to convert to speech
            
        Returns:
            bytes: Audio data (WAV format), or None if text is empty/whitespace-only
        """
        logger.info(f"TTS synthesize called with text: '{text}', type: {type(text)}, length: {len(str(text))}")
        
        # Handle empty or whitespace-only text as "don't say anything"
        if not text or not str(text).strip():
            logger.info("Empty or whitespace-only text provided - returning None (silent)")
            return None
            
        if not self.is_initialized():
            logger.error("ERROR: TTS engine not initialized in synthesize method!")
            logger.error(f"Pipeline exists: {self.pipeline is not None}, Initialized flag: {self.initialized}")
            return None
            
        logger.info(f"TTS engine initialized, proceeding with synthesis")

        # Split text into chunks if needed
        chunks = self._preprocess_text(str(text).strip())
        
        if not chunks:
            logger.info("Text preprocessing returned no chunks")
            return None
            
        logger.info(f"Processing {len(chunks)} text chunks for TTS")
        
        # If only one chunk, optimize for single chunk case
        if len(chunks) == 1:
            chunk = chunks[0].strip()
            if not chunk:
                logger.info("Single chunk is empty")
                return None
                
            logger.info(f"Generating audio for single chunk: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")
            # Generate directly as WAV for single chunk
            audio_data = await self._generate_audio(chunk)
            
            if audio_data:
                logger.info(f"Generated {len(audio_data)} bytes of audio for single chunk")
                return audio_data
            else:
                logger.info("Failed to generate audio for single chunk")
                return None
        
        # Multiple chunks - use optimized numpy array processing
        audio_arrays = []
        
        # Process each chunk and collect numpy arrays
        for i, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if not chunk:
                logger.info(f"Skipping empty chunk {i+1}/{len(chunks)}")
                continue
                
            logger.info(f"Generating audio for chunk {i+1}/{len(chunks)}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")
            audio_np = await self._generate_audio_numpy(chunk)
            
            if audio_np is not None:
                audio_arrays.append(audio_np)
                logger.info(f"Generated {len(audio_np)} samples for chunk {i+1}")
            else:
                logger.info(f"Failed to generate audio for chunk {i+1}")
        
        if not audio_arrays:
            logger.info("No audio arrays were generated")
            return None
        
        # Combine all numpy arrays efficiently
        combined_audio = self._combine_audio_numpy(audio_arrays)
        
        if combined_audio is None:
            logger.error("Failed to combine audio arrays")
            return None
        
        # Convert final combined array to WAV only once
        try:
            buffer = io.BytesIO()
            sf.write(buffer, combined_audio, 24000, format="WAV", subtype="PCM_16")
            buffer.seek(0)
            final_audio = buffer.read()
            
            logger.info(f"Final combined audio size: {len(final_audio)} bytes")
            return final_audio
            
        except Exception as e:
            logger.error(f"Error converting final audio to WAV: {e}")
            return None
    
    @traceable(run_type="chain", name="TTS_Synthesize_Speech_Streaming")
    async def synthesize_streaming(self, text):
        """Convert text to speech and yield audio for each sentence as it's generated
        
        Args:
            text: Text to convert to speech
            
        Yields:
            bytes: Audio data chunks (WAV format), one per sentence or natural speech pause
        """
        logger.info(f"TTS streaming synthesize called with text: '{text}', type: {type(text)}, length: {len(str(text))}")
        
        # Handle empty or whitespace-only text as "don't say anything"
        if not text or not str(text).strip():
            logger.info("Empty or whitespace-only text provided - yielding nothing (silent)")
            return
            
        if not self.is_initialized():
            logger.error("ERROR: TTS engine not initialized in synthesize_streaming method!")
            logger.error(f"Pipeline exists: {self.pipeline is not None}, Initialized flag: {self.initialized}")
            return
            
        logger.info(f"TTS engine initialized, proceeding with streaming synthesis")

        # Split text into sentences or natural speech pauses
        sentences = self._preprocess_text(str(text).strip())
        
        if not sentences:
            logger.info("Text preprocessing returned no sentences")
            return
            
        logger.info(f"Processing {len(sentences)} sentences for streaming TTS")
        
        # Process each sentence individually and yield audio as soon as it's ready
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                logger.info(f"Skipping empty sentence {i+1}/{len(sentences)}")
                continue
                
            # Process this sentence - log with abbreviated content to avoid cluttering logs
            max_log_chars = 50
            log_text = sentence[:max_log_chars] + ('...' if len(sentence) > max_log_chars else '')
            logger.info(f"Generating audio for sentence {i+1}/{len(sentences)}: '{log_text}'")
            
            # Generate audio for this specific sentence
            try:
                # Generate audio for this sentence - async to avoid blocking
                sentence_start = time.time()
                audio_data = await self._generate_audio(sentence)
                
                if audio_data:
                    logger.info(f"Yielding {len(audio_data)} bytes of audio for sentence {i+1}/{len(sentences)} ({time.time() - sentence_start:.2f}s)")
                    yield audio_data
                else:
                    logger.info(f"Failed to generate audio for sentence {i+1}/{len(sentences)}")
            except Exception as e:
                logger.error(f"Error generating audio for sentence {i+1}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next sentence on error
    
    def _combine_audio_chunks(self, audio_chunks):
        """Legacy method for combining WAV chunks - kept for backward compatibility
        
        Args:
            audio_chunks: List of WAV audio data bytes
            
        Returns:
            bytes: Combined audio data
        """
        if not audio_chunks:
            logger.error("No audio chunks to combine")
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
                    # Use io.BytesIO to read from bytes without writing to disk
                    with io.BytesIO(chunk) as buf:
                        # Read data using soundfile
                        data, sr = sf.read(buf)
                        
                        # Store first sample rate
                        if i == 0:
                            sample_rate = sr
                            
                        # Ensure consistent sample rate
                        if sr != sample_rate:
                            logger.warning(f"Warning: Chunk {i+1} has different sample rate ({sr} vs {sample_rate})")
                            
                        # Add to list
                        wav_data.append(data)
                        logger.info(f"Read chunk {i+1}: {len(data)} samples at {sr}Hz")
                        
                except Exception as e:
                    logger.error(f"Error reading chunk {i+1}: {e}")
                    
            # Check if we got any data
            if not wav_data:
                logger.error("No chunks could be read")
                return None
                
            # Concatenate audio data
            logger.debug(f"Concatenating {len(wav_data)} audio chunks...")
            combined = np.concatenate(wav_data)
            
            # Write to buffer
            buffer = io.BytesIO()
            sf.write(buffer, combined, sample_rate, format="WAV", subtype="PCM_16")
            buffer.seek(0)
            
            return buffer.read()
            
        except Exception as e:
            logger.error(f"Error combining audio chunks: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def get_available_voices(self) -> List[str]:
        """Get a list of available voices"""
        # For now, we only have a few fixed voice options
        return [
            "af_heart",    # American female heart
            "am_casual",   # American male casual
            "af_casual",   # American female casual
            "bm_casual",   # British male casual
            "bf_casual",   # British female casual
        ]
        
    def is_initialized(self) -> bool:
        """Check if the TTS engine is properly initialized"""
        return self.initialized and self.pipeline is not None 