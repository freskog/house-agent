# Audio Server Module

The Audio Server module provides WebSocket-based voice interaction capabilities, including:

- Real-time streaming audio processing
- Voice Activity Detection (VAD)
- Speech-to-Text using Whisper
- Text-to-Speech using Kokoro

## Quick Start

### Install Dependencies

```bash
# Install PyTorch (for Silero VAD)
pip install torch torchaudio

# Install PyWhisperCPP (for Whisper transcription)
pip install pywhispercpp

# Install Kokoro (for TTS)
pip install kokoro

# Install other requirements
pip install websockets pyaudio numpy soundfile pydantic
```

### Starting the Server

```bash
python -m audio.server
```

### Command-line Options

```
--host              Host to bind to (default: localhost)
--port              Port to listen on (default: 8765)
--vad-threshold     Voice detection threshold 0.0-1.0 (default: 0.3)
--whisper-model     Whisper model to use (default: medium-q5_0)
                    Options: tiny, base, small, medium, large-v3
--tts-voice         TTS voice to use (default: af_heart)
--use-coreml        Use CoreML acceleration on Apple Silicon
--verbose           Enable verbose logging
```

Example:
```bash
python -m audio.server --host=0.0.0.0 --port=8765 --whisper-model=small --use-coreml
```

### Testing with the Edge Device Simulator

Run the client to simulate an edge device:

```bash
python -m audio.client
```

**Important**: Use headphones to prevent audio feedback!

## Architecture

The audio server uses a WebSocket protocol for bidirectional communication with clients. The key components are:

1. **WebSocket Server** (`websocket.py`): Handles client connections and coordinates audio processing
2. **Voice Activity Detection** (`vad.py`): Detects speech in audio streams using Silero VAD
3. **Transcription** (`transcribe.py`): Converts speech to text using Whisper via PyWhisperCPP
4. **TTS Engine** (`tts.py`): Converts text to speech using Kokoro

## Protocol

The WebSocket protocol uses a standardized message format with these types:

- `audio_stream`: Client sends audio data to server
- `audio_playback`: Server sends audio data to client for playback
- `status`: Server sends status updates (VAD, transcription)
- `error`: Server sends error messages

Audio formats:
- **Input audio**: Always raw PCM, 16-bit, mono, 16kHz
- **Output audio**: Always WAV format

## Configuration

The server components can be configured:

```python
from audio.server.websocket import AudioServer
from audio.server.vad import VADConfig
from audio.server.transcribe import TranscriptionConfig
from audio.server.tts import TTSConfig

# Configure components
vad_config = VADConfig(threshold=0.5)
transcription_config = TranscriptionConfig(model_name="large-v3")
tts_config = TTSConfig(voice="am_bear")

# Create server with configuration
server = AudioServer(
    vad_config=vad_config,
    transcription_config=transcription_config,
    tts_config=tts_config
)
```

## Client Implementation

Clients must implement the WebSocket protocol. The main flow is:

1. Connect to the WebSocket server
2. Stream audio data in 16-bit, mono, 16kHz PCM format
3. Receive status updates (speech detection)
4. Receive transcription results
5. Receive audio responses for playback

See `audio/client/simple_client.py` for a reference implementation. 