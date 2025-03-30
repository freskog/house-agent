# House Agent

A voice-controlled home automation assistant with CoreML-accelerated speech recognition.

## Features

- Audio recording from your microphone
- Transcription using Whisper.cpp with CoreML acceleration on Apple Silicon
- Detailed transcription metrics and benchmarking
- Multiple Whisper model support (tiny, base, small, medium, large-v3)

## Requirements

- macOS (preferably with Apple Silicon for CoreML acceleration)
- Python 3.8+
- PyAudio
- pywhispercpp

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/house-agent.git
cd house-agent
```

2. Run the setup script:
```bash
chmod +x setup_speech.sh
./setup_speech.sh
```

The setup script will:
- Install necessary system dependencies via Homebrew
- Install required Python packages
- Configure CoreML acceleration on Apple Silicon

## Usage

Run the speech test utility:
```bash
python speech_test.py
```

### Available Commands

- Interactive mode (default): `python speech_test.py`
- Batch processing: `python speech_test.py --batch=speech_samples`
- List available models: `python speech_test.py --list-models`

### Interactive Mode Features

The interactive mode allows you to:
1. Record audio from your microphone
2. Transcribe recorded audio using Whisper
3. Save audio samples for future testing
4. Load and transcribe previously recorded audio files
5. Change Whisper model sizes for different accuracy/speed tradeoffs

## Performance

With CoreML acceleration on Apple Silicon:
- Base model: ~0.05x real-time (20x faster than real-time)
- Large-v3 model: ~0.2-0.4x real-time (2.5-5x faster than real-time)

## License

MIT License 