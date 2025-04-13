"""
Main entry point for running the audio client as a module.
Run with: python -m audio.client
"""

import asyncio
import sys
from audio.client.simple_client import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient stopped by user")
        sys.exit(0) 