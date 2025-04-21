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
        """Preprocess text for TTS - sentence splitting, normalization etc.
        
        This method splits text into natural speech units (sentences, phrases at commas, etc.)
        to allow real-time streaming of audio as soon as each segment is processed.
        
        Returns smaller chunks for faster initial processing, especially for long texts.
        """
        try:
            # First protect common abbreviations from being split incorrectly
            protected_text = text
            abbreviations = ["Mr.", "Mrs.", "Dr.", "Ph.D.", "M.D.", "B.A.", "B.S.", "M.S.", "i.e.", "e.g.", "U.S.A.", "U.K.", "St."]
            
            for abbr in abbreviations:
                protected_text = protected_text.replace(abbr, abbr.replace(".", "{{DOT}}"))
            
            # Split on sentence boundaries first (periods, exclamation marks, question marks)
            sentence_boundaries = r'(?<=[.!?])\s+'
            sentences = re.split(sentence_boundaries, protected_text)
            
            # Further split long sentences at natural pauses (commas, semicolons, colons, dashes)
            final_segments = []
            natural_pause_boundaries = r'(?<=[,;:\-–—])\s+'
            
            # Use smaller chunks for faster initial output
            # Shorter segments = faster first-chunk time
            max_segment_length = 80  # Reduced from 120 for faster processing
            
            for sentence in sentences:
                # Restore the protected abbreviations
                sentence = sentence.replace("{{DOT}}", ".")
                
                if len(sentence) <= max_segment_length:
                    # If sentence is a reasonable length, keep as is
                    final_segments.append(sentence)
                else:
                    # For long sentences, split at natural pause points
                    pause_segments = re.split(natural_pause_boundaries, sentence)
                    
                    # If split resulted in reasonable segments, use them
                    if all(len(seg) <= max_segment_length for seg in pause_segments):
                        final_segments.extend(pause_segments)
                    else:
                        # For still-too-long segments, use a fallback approach - split by length
                        for segment in pause_segments:
                            if len(segment) <= max_segment_length:
                                final_segments.append(segment)
                            else:
                                # Use a simple chunk-by-length approach for very long text with no natural breaks
                                chunks = []
                                for i in range(0, len(segment), max_segment_length):
                                    chunk = segment[i:i+max_segment_length]
                                    # Try to avoid splitting words if possible
                                    if i+max_segment_length < len(segment) and not segment[i+max_segment_length].isspace():
                                        # Find the last space in this chunk
                                        last_space = chunk.rfind(' ')
                                        if last_space > max_segment_length * 0.5:  # Only adjust if space is reasonably positioned
                                            chunks.append(chunk[:last_space])
                                            # Adjust start of next chunk
                                            i = i + last_space + 1
                                            continue
                                    chunks.append(chunk)
                                final_segments.extend(chunks)
            
            # Filter out empty segments and trim whitespace
            final_segments = [s.strip() for s in final_segments if s.strip()]
            
            # For very long texts (like reading out article content), prioritize the first few segments
            # This makes the system respond faster with the beginning of the content
            if len(final_segments) > 10:
                # Rearrange segments to prioritize the first few
                # Take first 3 segments, then every other segment until end
                # This helps initial chunks get generated and sent faster
                prioritized_segments = final_segments[:3]
                prioritized_segments.extend(final_segments[3::2])  # Every other segment after the first 3
                prioritized_segments.extend(final_segments[4::2])  # The remaining segments
                final_segments = prioritized_segments
            
            print(f"Split text into {len(final_segments)} segments for natural speech flow")
            return final_segments if final_segments else [text]
        
        except Exception as e:
            print(f"Error during text preprocessing: {e}")
            import traceback
            traceback.print_exc()
            # Safest fallback is to return the whole text as one chunk
            return [text]
    

    async def _generate_audio(self, text):
        """Generate audio from text using Kokoro pipeline
        
        Optimized for single sentences or small chunks of text.
        """
        if not self.pipeline:
            print("Kokoro pipeline not initialized")
            return None
            
        try:
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
                    print(f"Error in TTS pipeline: {e}")
                    return None
            
            # Run in thread pool with a timeout to prevent hanging
            try:
                audio_data = await asyncio.wait_for(
                    loop.run_in_executor(None, process_text),
                    timeout=10.0  # 10 second timeout per sentence
                )
            except asyncio.TimeoutError:
                print(f"Timeout generating audio for: '{text[:30]}...'")
                return None
            
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
            
    @traceable(run_type="chain", name="TTS_Synthesize_Speech")
    async def synthesize(self, text):
        """Convert text to speech and return audio data
        
        Args:
            text: Text to convert to speech
            
        Returns:
            bytes: Audio data (WAV format)
        """
        print(f"TTS synthesize called with text: '{text}', type: {type(text)}, length: {len(str(text))}")
        
        if not text:
            print("Empty text provided to synthesize method")
            return None
            
        if not self.is_initialized():
            print("ERROR: TTS engine not initialized in synthesize method!")
            print(f"Pipeline exists: {self.pipeline is not None}, Initialized flag: {self.initialized}")
            return None
            
        print(f"TTS engine initialized, proceeding with synthesis")

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
        
    async def synthesize_streaming(self, text):
        """Convert text to speech and yield audio for each sentence as it's generated
        
        Args:
            text: Text to convert to speech
            
        Yields:
            bytes: Audio data chunks (WAV format), one per sentence or natural speech pause
        """
        print(f"TTS streaming synthesize called with text: '{text}', type: {type(text)}, length: {len(str(text))}")
        
        if not text:
            print("Empty text provided to synthesize_streaming method")
            return
            
        if not self.is_initialized():
            print("ERROR: TTS engine not initialized in synthesize_streaming method!")
            print(f"Pipeline exists: {self.pipeline is not None}, Initialized flag: {self.initialized}")
            return
            
        print(f"TTS engine initialized, proceeding with streaming synthesis")

        # Split text into sentences or natural speech pauses
        sentences = self._preprocess_text(text)
        
        if not sentences:
            print("Text preprocessing returned no sentences")
            return
            
        print(f"Processing {len(sentences)} sentences for streaming TTS")
        
        # Process each sentence individually and yield audio as soon as it's ready
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                print(f"Skipping empty sentence {i+1}/{len(sentences)}")
                continue
                
            # Process this sentence - log with abbreviated content to avoid cluttering logs
            max_log_chars = 50
            log_text = sentence[:max_log_chars] + ('...' if len(sentence) > max_log_chars else '')
            print(f"Generating audio for sentence {i+1}/{len(sentences)}: '{log_text}'")
            
            # Generate audio for this specific sentence
            try:
                # Generate audio for this sentence - async to avoid blocking
                sentence_start = time.time()
                audio_data = await self._generate_audio(sentence)
                
                if audio_data:
                    print(f"Yielding {len(audio_data)} bytes of audio for sentence {i+1}/{len(sentences)} ({time.time() - sentence_start:.2f}s)")
                    yield audio_data
                else:
                    print(f"Failed to generate audio for sentence {i+1}/{len(sentences)}")
            except Exception as e:
                print(f"Error generating audio for sentence {i+1}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next sentence on error
    
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
                    # Use io.BytesIO to read from bytes without writing to disk
                    with io.BytesIO(chunk) as buf:
                        # Read data using soundfile
                        data, sr = sf.read(buf)
                        
                        # Store first sample rate
                        if i == 0:
                            sample_rate = sr
                            
                        # Ensure consistent sample rate
                        if sr != sample_rate:
                            print(f"Warning: Chunk {i+1} has different sample rate ({sr} vs {sample_rate})")
                            
                        # Add to list
                        wav_data.append(data)
                        print(f"Read chunk {i+1}: {len(data)} samples at {sr}Hz")
                        
                except Exception as e:
                    print(f"Error reading chunk {i+1}: {e}")
                    
            # Check if we got any data
            if not wav_data:
                print("No chunks could be read")
                return None
                
            # Concatenate audio data
            print(f"Concatenating {len(wav_data)} audio chunks...")
            combined = np.concatenate(wav_data)
            
            # Write to buffer
            buffer = io.BytesIO()
            sf.write(buffer, combined, sample_rate, format="WAV", subtype="PCM_16")
            buffer.seek(0)
            
            return buffer.read()
            
        except Exception as e:
            print(f"Error combining audio chunks: {e}")
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