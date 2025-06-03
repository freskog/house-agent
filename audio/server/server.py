"""
WebSocket Audio Server

This script starts a WebSocket server that provides:
- Voice Activity Detection (VAD)
- Speech-to-Text (Whisper)
- Text-to-Speech (Kokoro)

Usage:
    python -m audio.server [options]

Options:
    --host              Host to bind to (default: localhost)
    --port              Port to listen on (default: 8765)
    --vad-threshold     Voice detection threshold 0.0-1.0 (default: 0.3)
    --whisper-model     Whisper model to use (default: medium-q5_0)
                        Options: tiny, base, small, medium, large-v3
    --tts-voice         TTS voice to use (default: af_heart)
    --use-coreml        Use CoreML acceleration on Apple Silicon
    --verbose           Enable verbose logging
    --use-langsmith     Enable LangSmith tracing (disabled by default)
    --langsmith-api-key LangSmith API key for tracing
    --langsmith-project LangSmith project name for tracing

Example:
    python -m audio.server --host=0.0.0.0 --port=8765 --whisper-model=small --use-coreml
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
from typing import Optional, Dict, Any

from audio.server.websocket import AudioServer
from audio.server.vad import VADConfig
from audio.server.transcribe import TranscriptionConfig
from audio.server.tts import TTSConfig

# Configure LangSmith for tracing
os.environ["LANGSMITH_TRACING"] = os.environ.get("LANGSMITH_TRACING_V2", "false")
os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "voice-assistant")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('audio.server')

# Reduce websockets logging noise
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('websockets.server').setLevel(logging.WARNING)
logging.getLogger('websockets.protocol').setLevel(logging.WARNING)

async def default_transcription_callback(result):
    """Default response handler for transcription"""
    text = result.text.strip()
    logger.info(f"Received transcription: '{text}'")
    
    # Handle empty transcription
    if not text:
        return "I couldn't hear what you said. Could you please speak louder or try again?"
    
    # Return normal response for non-empty text
    return f"You said: {text}"

class ServerRunner:
    """Runner for the WebSocket audio server"""
    
    def __init__(self, host: str, port: int, 
                vad_threshold: float = 0.3,
                whisper_model: str = "medium-q5_0",
                tts_voice: str = "af_heart",
                use_coreml: bool = False):
        
        # Configure components
        vad_config = VADConfig(threshold=vad_threshold)
        transcription_config = TranscriptionConfig(
            model_name=whisper_model,
            use_coreml=use_coreml
        )
        tts_config = TTSConfig(voice=tts_voice)
        
        # Create server
        self.server = AudioServer(
            host=host,
            port=port,
            vad_config=vad_config,
            transcription_config=transcription_config,
            tts_config=tts_config,
            transcription_callback=default_transcription_callback
        )
        
        # Store configuration
        self.host = host
        self.port = port
        
    async def run(self):
        """Run the server"""
        # Set up signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            
        # Start server
        await self.server.start()
        logger.info(f"Server running at ws://{self.host}:{self.port}")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
            
    async def shutdown(self):
        """Shutdown the server gracefully"""
        logger.info("Shutting down server...")
        await self.server.stop()
        logger.info("Server stopped")
        
        # Stop the event loop
        loop = asyncio.get_running_loop()
        loop.stop()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="WebSocket Audio Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--vad-threshold", type=float, default=0.3, help="VAD threshold (0.0-1.0)")
    parser.add_argument("--whisper-model", default="medium-q5_0", help="Whisper model to use (tiny, base, small, medium, large-v3)")
    parser.add_argument("--tts-voice", default="af_heart", help="TTS voice to use")
    parser.add_argument("--use-coreml", action="store_true", help="Use CoreML acceleration on Apple Silicon (if available)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--use-langsmith", action="store_true", help="Enable LangSmith tracing (disabled by default)")
    parser.add_argument("--langsmith-api-key", help="LangSmith API key for tracing")
    parser.add_argument("--langsmith-project", help="LangSmith project name for tracing")
    
    return parser.parse_args()

async def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Configure LangSmith if specified
    if args.use_langsmith:
        if args.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = args.langsmith_api_key
            logger.info("LangSmith API key configured from arguments")
        
        if args.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = args.langsmith_project
            logger.info(f"LangSmith project set to: {args.langsmith_project}")
        os.environ["LANGSMITH_TRACING_V2"] = "true"
    else:
        os.environ["LANGSMITH_TRACING_V2"] = "false"
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Create configs
    vad_config = VADConfig(threshold=args.vad_threshold)
    
    transcription_config = TranscriptionConfig(
        model_name=args.whisper_model,
        use_coreml=args.use_coreml
    )
    
    tts_config = TTSConfig(voice=args.tts_voice)
    
    # Create and run server
    runner = ServerRunner(
        host=args.host,
        port=args.port,
        vad_threshold=args.vad_threshold,
        whisper_model=args.whisper_model,
        tts_voice=args.tts_voice,
        use_coreml=args.use_coreml
    )
    
    await runner.run()
    
if __name__ == "__main__":
    asyncio.run(main()) 