"""
Main entry point for running the audio server as a module.
Run with: python -m audio.server
"""

import asyncio
import sys
from audio.server.server import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0) 