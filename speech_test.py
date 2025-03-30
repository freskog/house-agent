#!/usr/bin/env python3
"""
Speech-to-Text Test Utility using pywhispercpp

This script allows you to:
1. Record audio from your microphone
2. Transcribe it using whisper.cpp (optimized for CoreML on Apple Silicon)
3. Save audio samples for future testing
"""

import os
import time
import wave
import argparse
import pyaudio
import tempfile
import platform
from datetime import datetime

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    # Ensure WHISPER_COREML is set
    if os.environ.get("WHISPER_COREML") != "1" and platform.system() == "Darwin" and platform.machine() == "arm64":
        print("Setting WHISPER_COREML=1 for this session")
        os.environ["WHISPER_COREML"] = "1"
except ImportError:
    pass

# Import pywhispercpp
try:
    from pywhispercpp.model import Model
    import pywhispercpp.constants as constants
    WHISPER_AVAILABLE = True
except ImportError:
    print("Warning: pywhispercpp not installed. Only recording will be available.")
    WHISPER_AVAILABLE = False

# Constants
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024

# Create directory for saving audio samples
SAMPLES_DIR = "speech_samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

class AudioRecorder:
    """Audio recorder that captures audio from the microphone"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, channels=CHANNELS, chunk=CHUNK):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.format = FORMAT
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.is_recording = False
        self.stream = None
        
    def start_recording(self, duration=None):
        """Start recording audio"""
        self.frames = []
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._callback if duration is None else None
        )
        
        self.is_recording = True
        
        if duration is None:
            self.stream.start_stream()
            print("Recording started. Press Ctrl+C to stop...")
        else:
            # Record for a fixed duration
            print(f"Recording for {duration} seconds...")
            for _ in range(0, int(self.sample_rate / self.chunk * duration)):
                data = self.stream.read(self.chunk)
                self.frames.append(data)
            self.stop_recording()
            
    def _callback(self, in_data, frame_count, time_info, status):
        """Callback for audio recording"""
        self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    def stop_recording(self):
        """Stop recording audio"""
        if self.stream:
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
        
        self.is_recording = False
        print("Recording stopped.")
        
    def save_recording(self, filename):
        """Save the recorded audio to a WAV file"""
        if not self.frames:
            print("No audio data to save.")
            return None
            
        # Create full path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not filename.endswith('.wav'):
            filename = f"{filename}.wav"
        
        full_path = os.path.join(SAMPLES_DIR, f"{timestamp}_{filename}")
        
        # Save the audio data
        with wave.open(full_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
            
        print(f"Audio saved to {full_path}")
        return full_path
    
    def get_temp_recording(self):
        """Save the recording to a temporary file and return the path"""
        if not self.frames:
            print("No audio data to save.")
            return None
            
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Save the audio data to the temporary file
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
            
        return temp_path
        
    def close(self):
        """Clean up resources"""
        self.audio.terminate()


class WhisperTranscriber:
    """Transcribes speech using pywhispercpp"""
    
    # Define a list of known valid models - this may not be complete but covers common options
    KNOWN_MODELS = [
        "tiny", "tiny.en", "tiny-q5_1", "tiny.en-q5_1", "tiny.en-q8_0",
        "base", "base.en", "base-q5_1", "base.en-q5_1",  
        "small", "small.en", "small-q5_1", "small.en-q5_1",
        "medium", "medium.en", "medium-q5_0", "medium.en-q5_0",
        "large-v1", "large-v2", "large-v2-q5_0", 
        "large-v3", "large-v3-q5_0", "large-v3-turbo", "large-v3-turbo-q5_0"
    ]
    
    def __init__(self, model_size="base", use_coreml=True):
        if not WHISPER_AVAILABLE:
            raise ImportError("pywhispercpp is not available")

        # Validate model against known models
        if model_size not in self.KNOWN_MODELS:
            print(f"Warning: '{model_size}' is not in the list of known models.")
            print(f"Defaulting to 'base' model...")
            model_size = "base"  # Fall back to base model
            
        # Setup parameters - these will be passed to transcribe(), not model initialization
        self.params = {}
        self.params["language"] = "en"  # auto-detect
        self.params["translate"] = False
        
        # Initialize the model
        print(f"Initializing whisper model: {model_size}")
        
        # CoreML usage is set during model initialization as an environment variable,
        # not as a parameter. The setup_speech.sh script handles this.
        if use_coreml and self._is_apple_silicon():
            print("Using CoreML acceleration...")
            # CoreML is already enabled via environment variable in setup script
            # We don't need to do anything here, just inform the user
        
        # Create the model - CoreML is enabled at compile time or via env variables
        self.model_size = model_size
        try:
            self.model = Model(model=model_size)
            print("Model initialization complete!")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model '{model_size}': {e}")
    
    def _is_apple_silicon(self):
        """Check if running on Apple Silicon"""
        return platform.system() == "Darwin" and platform.machine() == "arm64"
    
    def transcribe_file(self, filepath):
        """Transcribe audio from a file"""
        if not os.path.exists(filepath):
            return {"text": "File not found", "processing_time": 0, "audio_length": 0, "segments": []}
        
        # Get audio length
        with wave.open(filepath, 'rb') as wf:
            audio_length = wf.getnframes() / wf.getframerate()
        
        try:
            # Transcribe the audio - pass params to transcribe method
            start_time = time.time()
            segments = self.model.transcribe(filepath, **self.params)
            transcription_time = time.time() - start_time
            
            # Get only the text from the result (pywhispercpp returns segments)
            text = ""
            segment_list = []
            for segment in segments:
                text += segment.text + " "
                # Store segment data with timestamps if available
                segment_data = {
                    "text": segment.text,
                    "start": getattr(segment, "t0", 0) if hasattr(segment, "t0") else getattr(segment, "start", 0),
                    "end": getattr(segment, "t1", 0) if hasattr(segment, "t1") else getattr(segment, "end", 0)
                }
                segment_list.append(segment_data)
            
            return {
                "text": text.strip(),
                "processing_time": transcription_time,
                "audio_length": audio_length,
                "segments": segment_list
            }
        except Exception as e:
            print(f"Error transcribing file: {e}")
            return {
                "text": f"Error transcribing file: {e}",
                "processing_time": 0,
                "audio_length": audio_length,
                "segments": []
            }


def interactive_mode():
    """Run an interactive session for recording and transcription"""
    recorder = AudioRecorder()
    
    try:
        # Try to initialize with CoreML on Apple Silicon
        transcriber = WhisperTranscriber(model_size="base", use_coreml=True)
        transcription_available = True
    except Exception as e:
        print(f"Error initializing transcriber: {e}")
        transcription_available = False
    
    try:
        while True:
            print("\n--- Speech Test Utility ---")
            print("1. Record audio (press Ctrl+C to stop)")
            print("2. Record audio for specific duration")
            if transcription_available:
                print("3. Transcribe last recording")
                print("4. Load and transcribe WAV file")
                print("5. Change Whisper model size")
            print("0. Exit")
            
            choice = input("\nSelect an option: ")
            
            if choice == "1":
                try:
                    recorder.start_recording()
                    while recorder.is_recording:
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    recorder.stop_recording()
                
                filename = input("Enter a name to save the recording (or press Enter to skip): ")
                if filename:
                    saved_path = recorder.save_recording(filename)
                    if saved_path and transcription_available:
                        transcribe_now = input("Transcribe now? (y/n): ")
                        if transcribe_now.lower() == 'y':
                            print("Transcribing...")
                            result = transcriber.transcribe_file(saved_path)
                            print("\n----- Transcription Results -----")
                            print(f"Text: {result['text']}")
                            print(f"\nAudio Length: {result['audio_length']:.2f} seconds")
                            print(f"Processing Time: {result['processing_time']:.2f} seconds")
                            print(f"Real-time Factor: {result['processing_time'] / result['audio_length']:.2f}x")
                            
                            if result['segments'] and len(result['segments']) > 1:
                                print("\nDetailed Segments:")
                                for i, segment in enumerate(result['segments']):
                                    print(f"  [{segment['start']:.1f}s → {segment['end']:.1f}s] {segment['text']}")
                else:
                    # Use a temporary file for transcription without saving
                    if transcription_available:
                        transcribe_now = input("Transcribe without saving? (y/n): ")
                        if transcribe_now.lower() == 'y':
                            temp_path = recorder.get_temp_recording()
                            if temp_path:
                                print("Transcribing...")
                                result = transcriber.transcribe_file(temp_path)
                                print("\n----- Transcription Results -----")
                                print(f"Text: {result['text']}")
                                print(f"\nAudio Length: {result['audio_length']:.2f} seconds")
                                print(f"Processing Time: {result['processing_time']:.2f} seconds")
                                print(f"Real-time Factor: {result['processing_time'] / result['audio_length']:.2f}x")
                                
                                if result['segments'] and len(result['segments']) > 1:
                                    print("\nDetailed Segments:")
                                    for i, segment in enumerate(result['segments']):
                                        print(f"  [{segment['start']:.1f}s → {segment['end']:.1f}s] {segment['text']}")
                                # Delete the temporary file
                                os.unlink(temp_path)
                    
            elif choice == "2":
                try:
                    duration = float(input("Enter duration in seconds: "))
                    recorder.start_recording(duration=duration)
                    
                    filename = input("Enter a name to save the recording (or press Enter to skip): ")
                    if filename:
                        saved_path = recorder.save_recording(filename)
                        if saved_path and transcription_available:
                            transcribe_now = input("Transcribe now? (y/n): ")
                            if transcribe_now.lower() == 'y':
                                print("Transcribing...")
                                result = transcriber.transcribe_file(saved_path)
                                print("\n----- Transcription Results -----")
                                print(f"Text: {result['text']}")
                                print(f"\nAudio Length: {result['audio_length']:.2f} seconds")
                                print(f"Processing Time: {result['processing_time']:.2f} seconds")
                                print(f"Real-time Factor: {result['processing_time'] / result['audio_length']:.2f}x")
                                
                                if result['segments'] and len(result['segments']) > 1:
                                    print("\nDetailed Segments:")
                                    for i, segment in enumerate(result['segments']):
                                        print(f"  [{segment['start']:.1f}s → {segment['end']:.1f}s] {segment['text']}")
                    else:
                        # Use a temporary file for transcription without saving
                        if transcription_available:
                            transcribe_now = input("Transcribe without saving? (y/n): ")
                            if transcribe_now.lower() == 'y':
                                temp_path = recorder.get_temp_recording()
                                if temp_path:
                                    print("Transcribing...")
                                    result = transcriber.transcribe_file(temp_path)
                                    print("\n----- Transcription Results -----")
                                    print(f"Text: {result['text']}")
                                    print(f"\nAudio Length: {result['audio_length']:.2f} seconds")
                                    print(f"Processing Time: {result['processing_time']:.2f} seconds")
                                    print(f"Real-time Factor: {result['processing_time'] / result['audio_length']:.2f}x")
                                    
                                    if result['segments'] and len(result['segments']) > 1:
                                        print("\nDetailed Segments:")
                                        for i, segment in enumerate(result['segments']):
                                            print(f"  [{segment['start']:.1f}s → {segment['end']:.1f}s] {segment['text']}")
                                    # Delete the temporary file
                                    os.unlink(temp_path)
                except ValueError:
                    print("Invalid duration. Please enter a number.")
            
            elif choice == "3" and transcription_available:
                # Get the last recording path from the sample directory
                files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith('.wav')])
                if not files:
                    print("No recordings found.")
                    continue
                
                last_recording = os.path.join(SAMPLES_DIR, files[-1])
                print(f"Transcribing last recording: {files[-1]}...")
                result = transcriber.transcribe_file(last_recording)
                
                print("\n----- Transcription Results -----")
                print(f"Text: {result['text']}")
                print(f"\nAudio Length: {result['audio_length']:.2f} seconds")
                print(f"Processing Time: {result['processing_time']:.2f} seconds")
                print(f"Real-time Factor: {result['processing_time'] / result['audio_length']:.2f}x")
                
                if result['segments'] and len(result['segments']) > 1:
                    print("\nDetailed Segments:")
                    for i, segment in enumerate(result['segments']):
                        print(f"  [{segment['start']:.1f}s → {segment['end']:.1f}s] {segment['text']}")
                    
            elif choice == "4" and transcription_available:
                print(f"Available recordings in {SAMPLES_DIR}:")
                files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith('.wav')])
                
                if not files:
                    print("No recordings found.")
                    continue
                    
                for i, file in enumerate(files):
                    print(f"{i+1}. {file}")
                    
                try:
                    file_idx = int(input("Enter file number to transcribe (or 0 to cancel): ")) - 1
                    if file_idx < 0:
                        continue
                        
                    filepath = os.path.join(SAMPLES_DIR, files[file_idx])
                    print("Transcribing...")
                    result = transcriber.transcribe_file(filepath)
                    
                    print("\n----- Transcription Results -----")
                    print(f"Text: {result['text']}")
                    print(f"\nAudio Length: {result['audio_length']:.2f} seconds")
                    print(f"Processing Time: {result['processing_time']:.2f} seconds")
                    print(f"Real-time Factor: {result['processing_time'] / result['audio_length']:.2f}x")
                    
                    if result['segments'] and len(result['segments']) > 1:
                        print("\nDetailed Segments:")
                        for i, segment in enumerate(result['segments']):
                            print(f"  [{segment['start']:.1f}s → {segment['end']:.1f}s] {segment['text']}")
                except (ValueError, IndexError):
                    print("Invalid selection.")
            
            elif choice == "5" and transcription_available:
                # Updated model options with correct version naming
                print("\nAvailable model sizes:")
                print("1. tiny      - Fastest, least accurate")
                print("2. base      - Fast, decent accuracy (default)")
                print("3. small     - Balanced speed/accuracy")
                print("4. medium    - Good accuracy, slower")
                print("5. large-v3  - Best accuracy, slowest")
                print("6. List all models")
                
                try:
                    model_choice = int(input("\nSelect model size (or 0 to cancel): "))
                    if model_choice == 0:
                        continue
                        
                    if model_choice == 6:
                        # Display all available models
                        list_available_models()
                        
                        # Ask for specific model name
                        print("\nEnter the exact model name from the list above:")
                        selected_size = input("Model name: ").strip()
                        if not selected_size:
                            continue
                    else:
                        # Standard model selection
                        # Updated model names to match actual available models
                        sizes = ["tiny", "base", "small", "medium", "large-v3"]
                        if 1 <= model_choice <= len(sizes):
                            selected_size = sizes[model_choice - 1]
                        else:
                            print("Invalid selection.")
                            continue
                    
                    print(f"Changing model to {selected_size}...")
                    
                    try:
                        # Re-initialize transcriber with new model
                        transcriber = WhisperTranscriber(model_size=selected_size, use_coreml=True)
                        print(f"Model changed to {selected_size}")
                    except Exception as e:
                        print(f"Error initializing {selected_size} model: {e}")
                        print("Try using the --list-models option to see all valid models.")
                except ValueError:
                    print("Invalid selection.")
                    
            elif choice == "0":
                break
                
            else:
                print("Invalid option. Please try again.")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        recorder.close()


def batch_test_mode(directory):
    """Test transcription on all WAV files in a directory"""
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return
        
    # Get all WAV files in the directory
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    if not wav_files:
        print(f"No WAV files found in {directory}")
        return
        
    # Initialize transcriber
    try:
        transcriber = WhisperTranscriber()
    except Exception as e:
        print(f"Error initializing transcriber: {e}")
        return
        
    # Process each file
    results = []
    for file in wav_files:
        filepath = os.path.join(directory, file)
        print(f"Processing {file}...")
        
        result = transcriber.transcribe_file(filepath)
        rtf = result["processing_time"] / result["audio_length"]
        
        print(f"\n----- Transcription Results: {file} -----")
        print(f"Text: {result['text']}")
        print(f"Audio Length: {result['audio_length']:.2f} seconds")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        print(f"Real-time Factor: {rtf:.2f}x")
        
        if result['segments'] and len(result['segments']) > 1:
            print("\nDetailed Segments:")
            for i, segment in enumerate(result['segments']):
                print(f"  [{segment['start']:.1f}s → {segment['end']:.1f}s] {segment['text']}")
        
        print("-" * 50)
        
        results.append({
            "file": file,
            "text": result["text"],
            "audio_length": result["audio_length"],
            "processing_time": result["processing_time"],
            "rtf": rtf
        })
    
    # Calculate and print summary statistics
    if results:
        rtfs = [r["rtf"] for r in results]
        audio_lengths = [r["audio_length"] for r in results]
        processing_times = [r["processing_time"] for r in results]
        
        print("\nBatch Test Summary:")
        print(f"Files processed: {len(results)}")
        print(f"Total audio length: {sum(audio_lengths):.2f} seconds")
        print(f"Total processing time: {sum(processing_times):.2f} seconds")
        print(f"Average RTF: {sum(rtfs) / len(rtfs):.2f}x")
        print(f"Min RTF: {min(rtfs):.2f}x")
        print(f"Max RTF: {max(rtfs):.2f}x")


def list_available_models():
    """List all known models available for pywhispercpp"""
    if not WHISPER_AVAILABLE:
        print("Warning: pywhispercpp not installed. Cannot list models.")
        return
        
    # Use the known models list from the WhisperTranscriber class
    models = WhisperTranscriber.KNOWN_MODELS
    
    print("\nAvailable Whisper Models:")
    print("-" * 25)
    
    # Group models by type
    model_types = {}
    for model in models:
        base_name = model.split('.')[0].split('-')[0]
        if base_name not in model_types:
            model_types[base_name] = []
        model_types[base_name].append(model)
        
    # Print models grouped by type
    for base_name, variants in sorted(model_types.items()):
        print(f"\n{base_name.upper()} models:")
        for variant in sorted(variants):
            print(f"  - {variant}")
            
    print("\nUsage notes:")
    print("  - Smaller models are faster but less accurate")
    print("  - Models with 'q5_0/q5_1' are quantized (smaller and faster)")
    print("  - Models with '.en' are English-only models")
    print("  - Models with '-v3' are newer versions (more accurate)") 
    print("  - 'turbo' variants are optimized for speed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech-to-Text Test Utility using pywhispercpp")
    parser.add_argument("--batch", metavar="DIR", help="Run batch test on all WAV files in directory")
    parser.add_argument("--list-models", action="store_true", help="List all available whisper models")
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
    elif args.batch:
        batch_test_mode(args.batch)
    else:
        interactive_mode() 