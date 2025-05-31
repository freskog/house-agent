"""
Spotify Web API integration module for house agent.

This module provides Spotify playback control through the Web API,
targeting a spotifyd device for audio output.
"""

from .client import SpotifyClient
from .tools import create_spotify_tools

__all__ = [
    'SpotifyClient',
    'create_spotify_tools'
]

__version__ = '1.0.0' 