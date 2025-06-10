#!/usr/bin/env python3
"""
Audio server entry point.

This script starts the audio server and keeps it running until interrupted.
"""

import asyncio
import signal
import argparse
import os
from audio.server import (
    AudioServer, 
    VADConfig, 
    TranscriptionConfig,
    TranscriptionResult
)

# Import our logging infrastructure
from utils.logging_config import setup_logging

# Set up logger for this module
logger = setup_logging(__name__)

async def demo_transcription_callback(transcription: TranscriptionResult) -> str:
    """Demo callback for transcriptions that simply echoes what was heard"""
    logger.info(f"🎤 Transcribed: {transcription.text}")
    
    # In a real application, this would process the text through a proper agent
    # and return a meaningful response
    
    # For demo purposes, just echo what was heard
    if transcription.text.strip():
        return f"You said: {transcription.text}"
    else:
        return "I didn't hear anything clearly. Could you repeat that?"

async def main(args):
    """Main entry point"""
    # Create configs
    vad_config = VADConfig(
        threshold=args.vad_threshold,
        min_speech_duration=args.min_speech_duration,
        min_silence_duration=args.min_silence_duration,
        sample_rate=args.sample_rate
    )
    
    transcription_config = TranscriptionConfig(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language if args.language != "auto" else None
    )
    
    # Create server - this will eagerly initialize all components
    logger.info(f"🚀 Starting audio server with {args.model} model on {args.device}...")
    server = AudioServer(
        host=args.host,
        port=args.port,
        vad_config=vad_config,
        transcription_config=transcription_config,
        transcription_callback=demo_transcription_callback,
        save_recordings=args.save_recordings
    )
    
    # Setup signal handlers
    loop = asyncio.get_running_loop()
    for s in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(
            s, lambda: asyncio.create_task(cleanup(server))
        )
    
    # Start server
    await server.start()
    logger.info(f"✅ Audio server started on ws://{args.host}:{args.port}")
    if args.save_recordings:
        logger.info("📁 Audio recording is ENABLED - Transcribed speech will be saved to 'recordings/' directory")
    else:
        logger.info("📁 Audio recording is disabled - Use --save-recordings flag to enable")
    logger.info("🛑 Press Ctrl+C to stop")
    
    # Keep running until interrupted
    try:
        # This will run forever until the process is stopped
        await asyncio.Future()
    finally:
        await cleanup(server)
        
async def cleanup(server):
    """Cleanup resources"""
    logger.info("🧹 Shutting down server...")
    await server.stop()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.get_event_loop().stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio WebSocket Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--vad-threshold", type=float, default=0.3, help="VAD threshold")
    parser.add_argument("--min-speech-duration", type=float, default=0.5, help="Minimum speech duration")
    parser.add_argument("--min-silence-duration", type=float, default=1.0, help="Minimum silence duration")
    parser.add_argument("--model", type=str, default="medium-q5_0", help="Whisper model to use")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run model on")
    parser.add_argument("--compute-type", type=str, default="int8", help="Compute type for model")
    parser.add_argument("--language", type=str, default="auto", help="Language for transcription")
    parser.add_argument("--save-recordings", action="store_true", dest="save_recordings", 
                        help="Save transcribed audio to disk")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user") 