"""
Spotify Web API client for controlling playback.
"""

import spotipy
from typing import Optional, Dict, List, Any
import logging

from .config import SpotifyConfig, get_spotify_config
from .auth import SpotifyAuthManager

logger = logging.getLogger(__name__)

class SpotifyClient:
    """Spotify Web API client for controlling music playback."""
    
    def __init__(self, config: Optional[SpotifyConfig] = None):
        """
        Initialize Spotify client.
        
        Args:
            config: Spotify configuration. If None, loads from environment.
        """
        self.config = config or get_spotify_config()
        self.auth_manager = SpotifyAuthManager(self.config)
        self._spotify: Optional[spotipy.Spotify] = None
        
    def _get_spotify_instance(self) -> Optional[spotipy.Spotify]:
        """Get authenticated Spotify instance."""
        if not self._spotify:
            token = self.auth_manager.get_access_token()
            if not token:
                return None
            self._spotify = spotipy.Spotify(auth=token)
        return self._spotify
        
    def authenticate(self, authorization_response_url: Optional[str] = None) -> bool:
        """
        Authenticate with Spotify.
        
        Args:
            authorization_response_url: Response URL from OAuth flow
            
        Returns:
            bool: True if authentication successful
        """
        if authorization_response_url:
            token = self.auth_manager.get_access_token(authorization_response_url)
            if token:
                self._spotify = spotipy.Spotify(auth=token)
                return True
        return self.auth_manager.is_authenticated()
        
    def get_auth_url(self) -> str:
        """Get authorization URL for OAuth flow."""
        return self.auth_manager.get_auth_url()
        
    def is_authenticated(self) -> bool:
        """Check if client is authenticated."""
        return self.auth_manager.is_authenticated()
        
    def get_devices(self) -> List[Dict[str, Any]]:
        """
        Get available Spotify devices.
        
        Returns:
            List of device information dicts
        """
        try:
            spotify = self._get_spotify_instance()
            if not spotify:
                return []
            
            devices = spotify.devices()
            return devices.get('devices', [])
        except Exception as e:
            logger.error(f"Error getting devices: {e}")
            return []
            
    def get_spotifyd_device_id(self) -> Optional[str]:
        """
        Get the device ID for the configured spotifyd device.
        
        Returns:
            str: Device ID if found, None otherwise
        """
        devices = self.get_devices()
        for device in devices:
            if device['name'] == self.config.spotifyd_device_name:
                return device['id']
        return None
        
    def get_current_playback(self) -> Optional[Dict[str, Any]]:
        """
        Get current playback information.
        
        Returns:
            Dict with current track and playback state, or None
        """
        try:
            spotify = self._get_spotify_instance()
            if not spotify:
                return None
                
            current = spotify.current_playback()
            return current
        except Exception as e:
            logger.error(f"Error getting current playback: {e}")
            return None
            
    def play(self, uri: Optional[str] = None, device_id: Optional[str] = None) -> bool:
        """
        Start or resume playback.
        
        Args:
            uri: Optional Spotify URI to play (track, album, playlist)
            device_id: Optional device ID. Uses spotifyd device if not specified.
            
        Returns:
            bool: True if successful
        """
        try:
            spotify = self._get_spotify_instance()
            if not spotify:
                return False
                
            target_device = device_id or self.get_spotifyd_device_id()
            
            if uri:
                spotify.start_playback(device_id=target_device, uris=[uri])
            else:
                spotify.start_playback(device_id=target_device)
            return True
        except Exception as e:
            logger.error(f"Error starting playback: {e}")
            return False
            
    def pause(self, device_id: Optional[str] = None) -> bool:
        """
        Pause playback.
        
        Args:
            device_id: Optional device ID. Uses spotifyd device if not specified.
            
        Returns:
            bool: True if successful
        """
        try:
            spotify = self._get_spotify_instance()
            if not spotify:
                return False
                
            target_device = device_id or self.get_spotifyd_device_id()
            spotify.pause_playback(device_id=target_device)
            return True
        except Exception as e:
            logger.error(f"Error pausing playback: {e}")
            return False
            
    def next_track(self, device_id: Optional[str] = None) -> bool:
        """
        Skip to next track.
        
        Args:
            device_id: Optional device ID. Uses spotifyd device if not specified.
            
        Returns:
            bool: True if successful
        """
        try:
            spotify = self._get_spotify_instance()
            if not spotify:
                return False
                
            target_device = device_id or self.get_spotifyd_device_id()
            spotify.next_track(device_id=target_device)
            return True
        except Exception as e:
            logger.error(f"Error skipping to next track: {e}")
            return False
            
    def previous_track(self, device_id: Optional[str] = None) -> bool:
        """
        Skip to previous track.
        
        Args:
            device_id: Optional device ID. Uses spotifyd device if not specified.
            
        Returns:
            bool: True if successful
        """
        try:
            spotify = self._get_spotify_instance()
            if not spotify:
                return False
                
            target_device = device_id or self.get_spotifyd_device_id()
            spotify.previous_track(device_id=target_device)
            return True
        except Exception as e:
            logger.error(f"Error skipping to previous track: {e}")
            return False
            
    def set_volume(self, volume: int, device_id: Optional[str] = None) -> bool:
        """
        Set playback volume.
        
        Args:
            volume: Volume level (0-100)
            device_id: Optional device ID. Uses spotifyd device if not specified.
            
        Returns:
            bool: True if successful
        """
        try:
            spotify = self._get_spotify_instance()
            if not spotify:
                return False
                
            if not 0 <= volume <= 100:
                raise ValueError("Volume must be between 0 and 100")
                
            target_device = device_id or self.get_spotifyd_device_id()
            spotify.volume(volume, device_id=target_device)
            return True
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
            return False
            
    def search(self, query: str, types: List[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Search for tracks, albums, artists, or playlists.
        
        Args:
            query: Search query
            types: List of types to search for ('track', 'album', 'artist', 'playlist')
            limit: Maximum number of results
            
        Returns:
            Dict with search results
        """
        try:
            spotify = self._get_spotify_instance()
            if not spotify:
                return {}
                
            search_types = types or ['track']
            results = spotify.search(query, limit=limit, type=','.join(search_types))
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {} 