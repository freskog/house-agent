"""
Main entry point for running the audio client as a module.
Run with: python -m audio.client
"""

import asyncio
import sys
from audio.client.simple_client import main
from utils.logging_config import setup_logging

# Setup logging for main module
logger = setup_logging(__name__)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🔌 Client stopped by user")
        sys.exit(0) 