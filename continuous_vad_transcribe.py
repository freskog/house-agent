#!/usr/bin/env python3
"""
Continuous VAD Recording and Transcription

This script:
1. Continuously listens for speech using VAD
2. When speech is detected, records until silence
3. Immediately transcribes the recording
4. Returns to listening mode
5. Repeats until user exits
"""

import os
import time
import wave
import tempfile
import platform
import pyaudio
import torch
import torchaudio
import signal
import sys
import threading
from datetime import datetime
from silero_vad import load_silero_vad, get_speech_timestamps
from pywhispercpp.model import Model

# Constants - Adjusted for better audio quality
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024  # Smaller chunk size for smoother processing
CHUNKS_PER_VAD = 2  # Reduced to lower latency
VAD_THRESHOLD = 0.2  # Lowered for better speech detection
VAD_MIN_SPEECH_DURATION = 0.25  # Shorter to catch brief speech
VAD_MIN_SILENCE_DURATION = 1.5  # Longer to avoid cutting off speech
MAX_RECORDING_DURATION = 30.0
OUTPUT_DIR = "recordings"
BUFFER_PADDING = 3  # Number of chunks to keep before speech is detected

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables for cleanup
p_audio = None
stream = None
running = True
transcriber = None

def cleanup():
    """Clean up resources when exiting"""
    global p_audio, stream, running
    print(f"\n{YELLOW}Cleaning up...{RESET}")
    
    running = False
    
    if stream is not None:
        try:
            if stream.is_active():
                stream.stop_stream()
            stream.close()
        except:
            pass
    
    if p_audio is not None:
        try:
            p_audio.terminate()
        except:
            pass

def signal_handler(sig, frame):
    """Handle Ctrl+C and other signals"""
    print(f"\n{YELLOW}Exiting...{RESET}")
    cleanup()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class Transcriber:
    """Handles transcription of audio files"""
    
    def __init__(self, model_size="large-v3-turbo", use_coreml=True):
        """Initialize the transcriber"""
        self.model_size = model_size
        self.use_coreml = use_coreml
        self.model = None
        
    def initialize(self):
        """Initialize the model"""
        print(f"{YELLOW}Initializing whisper model: {self.model_size}{RESET}")
        
        # Set CoreML usage for Apple Silicon
        if self.use_coreml and platform.system() == "Darwin" and platform.machine() == "arm64":
            os.environ["WHISPER_COREML"] = "1"
        
        try:
            self.model = Model(model=self.model_size)
            print(f"{GREEN}Model ready{RESET}")
            return True
        except Exception as e:
            print(f"{RED}Error initializing model: {e}{RESET}")
            return False
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio data directly"""
        if self.model is None:
            if not self.initialize():
                return "Failed to initialize model"
        
        try:
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Write audio data to temporary file
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p_audio.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data)
            
            # Get audio length
            with wave.open(temp_path, 'rb') as wf:
                audio_length = wf.getnframes() / wf.getframerate()
            
            # Transcribe
            print(f"{YELLOW}Transcribing...{RESET}")
            start_time = time.time()
            segments = self.model.transcribe(temp_path)
            
            # Extract text
            text = " ".join(segment.text for segment in segments)
            
            # Print results
            print(f"\n{BLUE}{BOLD}{text}{RESET}")
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
            
            return text
            
        except Exception as e:
            print(f"{RED}Error transcribing audio: {e}{RESET}")
            return f"Error: {e}"

def continuous_vad_transcribe():
    """Continuously listen for speech, transcribe, and repeat"""
    global p_audio, stream, running, transcriber
    
    print(f"\n{BOLD}=== Continuous VAD Recording and Transcription ==={RESET}")
    print(f"{YELLOW}Press Ctrl+C to exit{RESET}")
    
    # Initialize PyAudio
    p_audio = pyaudio.PyAudio()
    
    # Initialize VAD
    print(f"{YELLOW}Initializing VAD...{RESET}")
    try:
        vad_model = load_silero_vad()
        print(f"{GREEN}VAD ready{RESET}")
    except Exception as e:
        print(f"{RED}Error initializing VAD: {e}{RESET}")
        return
    
    # Initialize transcriber
    transcriber = Transcriber(model_size="medium-q5_0")
    if not transcriber.initialize():
        print(f"{RED}Failed to initialize transcriber. Exiting.{RESET}")
        return
    
    try:
        while running:
            print(f"\n{GREEN}🎤 Listening...{RESET}")
            
            frames = []
            buffer_frames = []  # Rolling buffer to capture audio before speech starts
            accumulated_chunks = []
            recording_active = False
            last_speech_time = None
            start_time = time.time()
            
            # Open audio stream with adjusted buffer size
            stream = p_audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_buffer_size=CHUNK * 8  # Larger input buffer to prevent overflow
            )
            
            # Listen for speech
            while running:
                try:
                    # Read audio data with lower latency
                    data = stream.read(CHUNK)
                    current_time = time.time()
                    
                    # Check if we've exceeded maximum recording duration
                    if current_time - start_time > MAX_RECORDING_DURATION:
                        if recording_active:
                            print(f"{YELLOW}Maximum recording duration reached{RESET}")
                        break
                    
                    # Keep a rolling buffer of recent frames
                    buffer_frames.append(data)
                    if len(buffer_frames) > BUFFER_PADDING:
                        buffer_frames.pop(0)
                    
                    # Add to accumulated chunks
                    accumulated_chunks.append(data)
                    
                    # Only run VAD when we have enough chunks
                    if len(accumulated_chunks) >= CHUNKS_PER_VAD:
                        # Combine chunks into a single tensor
                        combined_data = b''.join(accumulated_chunks)
                        
                        try:
                            # Convert to tensor for VAD
                            audio_tensor = torch.frombuffer(combined_data, dtype=torch.int16).float()
                            audio_tensor = audio_tensor / 32768.0  # Normalize to [-1, 1]
                            
                            # Ensure tensor is 1D
                            if len(audio_tensor.shape) > 1:
                                audio_tensor = audio_tensor.squeeze()
                            
                            # Check for speech using get_speech_timestamps
                            speech_timestamps = get_speech_timestamps(
                                audio_tensor,
                                vad_model,
                                sampling_rate=SAMPLE_RATE,
                                return_seconds=True,
                                threshold=VAD_THRESHOLD,
                                min_speech_duration_ms=VAD_MIN_SPEECH_DURATION * 1000
                            )
                            
                            if speech_timestamps:
                                # If we detect speech
                                if not recording_active:
                                    print(f"{RED}🔴 Recording...{RESET}")
                                    recording_active = True
                                    
                                    # Include buffer frames to catch the beginning of speech
                                    frames.extend(buffer_frames)
                                
                                # Add chunks to frames
                                frames.extend(accumulated_chunks)
                                last_speech_time = current_time
                                
                            elif recording_active:
                                # If we've been recording and there's silence
                                frames.extend(accumulated_chunks)
                                
                                # Check if silence has been long enough to stop recording
                                silence_duration = current_time - last_speech_time if last_speech_time else 0
                                if silence_duration > VAD_MIN_SILENCE_DURATION:
                                    print(f"{YELLOW}⏹️ Processing...{RESET}")
                                    break
                        
                        except Exception as e:
                            print(f"{RED}Error processing audio: {e}{RESET}")
                        
                        # Reset accumulated chunks but keep the last one for overlap
                        accumulated_chunks = [accumulated_chunks[-1]] if accumulated_chunks else []
                
                except IOError as e:
                    # Handle audio input errors more gracefully
                    print(f"{RED}Audio input error: {e}{RESET}")
                    time.sleep(0.1)  # Short pause before retrying
            
            # Close stream
            if stream:
                stream.stop_stream()
                stream.close()
                stream = None
            
            # Process recording if we have frames
            if frames and recording_active:
                audio_data = b''.join(frames)
                
                # Save recording with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.wav")
                
                try:
                    with wave.open(output_path, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(p_audio.get_sample_size(FORMAT))
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(audio_data)
                    
                    # Transcribe the audio immediately
                    transcriber.transcribe_audio(audio_data)
                except Exception as e:
                    print(f"{RED}Error saving or processing recording: {e}{RESET}")
            elif recording_active:
                print(f"{YELLOW}Recording too short, ignoring{RESET}")
    
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted by user{RESET}")
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}")
    finally:
        cleanup()

if __name__ == "__main__":
    continuous_vad_transcribe() 