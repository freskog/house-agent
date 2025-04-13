"""
WebSocket Audio Server

This script starts a WebSocket server that provides:
- Voice Activity Detection (VAD)
- Speech-to-Text (Whisper)
- Text-to-Speech (Kokoro)

Run with:
    python -m audio.server
"""

import asyncio
import argparse
import logging
import signal
import sys
from typing import Optional, Dict, Any

from audio.server.websocket import AudioServer
from audio.server.vad import VADConfig
from audio.server.transcribe import TranscriptionConfig
from audio.server.tts import TTSConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('audio.server')

async def default_transcription_callback(result):
    """Default response handler for transcription"""
    text = result.text.strip()
    logger.info(f"Received transcription: {text}")
    
    # Simple echo response
    return f"You said: {text}"

class ServerRunner:
    """Runner for the WebSocket audio server"""
    
    def __init__(self, host: str, port: int, 
                vad_threshold: float = 0.3,
                whisper_model: str = "medium-q5_0",
                tts_voice: str = "af_heart"):
        
        # Configure components
        vad_config = VADConfig(threshold=vad_threshold)
        transcription_config = TranscriptionConfig(model_name=whisper_model)
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
    parser.add_argument("--whisper-model", default="medium-q5_0", help="Whisper model to use")
    parser.add_argument("--tts-voice", default="af_heart", help="TTS voice to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()

async def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Create and run server
    runner = ServerRunner(
        host=args.host,
        port=args.port,
        vad_threshold=args.vad_threshold,
        whisper_model=args.whisper_model,
        tts_voice=args.tts_voice
    )
    
    await runner.run()
    
if __name__ == "__main__":
    asyncio.run(main()) 