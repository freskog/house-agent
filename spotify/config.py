"""
Configuration management for Spotify integration.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class SpotifyConfig:
    """Configuration for Spotify Web API integration."""
    
    client_id: str
    client_secret: str
    redirect_uri: str
    spotifyd_device_name: str = "spotifyd"
    scopes: list = None
    cache_path: str = os.path.expanduser("~/.cache/spotify_agent_cache")
    
    def __post_init__(self):
        """Set default scopes if not provided."""
        if self.scopes is None:
            self.scopes = [
                "user-read-playback-state",
                "user-modify-playback-state"
            ]

def get_spotify_config() -> SpotifyConfig:
    """
    Load Spotify configuration from environment variables.
    
    Returns:
        SpotifyConfig: Configuration object with credentials
        
    Raises:
        ValueError: If required environment variables are missing
    """
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET") 
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
    device_name = os.getenv("SPOTIFYD_DEVICE_NAME", "spotifyd")
    
    if not client_id:
        raise ValueError("SPOTIFY_CLIENT_ID environment variable is required")
    if not client_secret:
        raise ValueError("SPOTIFY_CLIENT_SECRET environment variable is required")
    if not redirect_uri:
        raise ValueError("SPOTIFY_REDIRECT_URI environment variable is required")
        
    return SpotifyConfig(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        spotifyd_device_name=device_name
    ) 