"""
Spotify Web API client for controlling playback.
"""

import spotipy
from typing import Optional, Dict, List, Any
import logging
import concurrent.futures

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
        # Use the modern oauth_manager approach for automatic token refresh
        self._spotify: Optional[spotipy.Spotify] = None
        
    async def _get_spotify_instance_async(self) -> Optional[spotipy.Spotify]:
        """Get authenticated Spotify instance with automatic token refresh (async version)."""
        if not self._spotify:
            # Use the SpotifyOAuth instance from auth_manager for automatic token refresh
            self._spotify = spotipy.Spotify(oauth_manager=self.auth_manager.auth_manager)
        return self._spotify
        
    def _get_spotify_instance(self) -> Optional[spotipy.Spotify]:
        """Get authenticated Spotify instance with automatic token refresh."""
        if not self._spotify:
            # Use the SpotifyOAuth instance from auth_manager for automatic token refresh
            self._spotify = spotipy.Spotify(oauth_manager=self.auth_manager.auth_manager)
        return self._spotify
        
    async def authenticate_async(self, authorization_response_url: Optional[str] = None) -> bool:
        """
        Authenticate with Spotify (async version).
        
        Args:
            authorization_response_url: Response URL from OAuth flow
            
        Returns:
            bool: True if authentication successful
        """
        try:
            # Use the async auth methods
            if authorization_response_url:
                token = await self.auth_manager.get_access_token_async(authorization_response_url)
                if token:
                    # Reset the spotify instance so it gets recreated with fresh auth
                    self._spotify = None
                    return True
            
            return await self.auth_manager.is_authenticated_async()
            
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return False
        
    def authenticate(self, authorization_response_url: Optional[str] = None) -> bool:
        """
        Authenticate with Spotify.
        
        Args:
            authorization_response_url: Response URL from OAuth flow
            
        Returns:
            bool: True if authentication successful
        """
        try:
            import asyncio
            
            # Check if we're in an async context and handle accordingly
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, use asyncio.to_thread
                async def async_auth():
                    if authorization_response_url:
                        token = await asyncio.to_thread(self.auth_manager.get_access_token, authorization_response_url)
                        if token:
                            # Reset the spotify instance so it gets recreated with fresh auth
                            self._spotify = None
                            return True
                    return await asyncio.to_thread(self.auth_manager.is_authenticated)
                
                # Run in thread to avoid blocking
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_auth())
                    return future.result(timeout=10)
                    
            except RuntimeError:
                # No running loop, we can call directly
                if authorization_response_url:
                    token = self.auth_manager.get_access_token(authorization_response_url)
                    if token:
                        # Reset the spotify instance so it gets recreated with fresh auth
                        self._spotify = None
                        return True
                return self.auth_manager.is_authenticated()
                
        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            return False
        
    def get_auth_url(self) -> str:
        """Get authorization URL for OAuth flow."""
        return self.auth_manager.get_auth_url()
        
    async def is_authenticated_async(self) -> bool:
        """Check if client is authenticated (async version)."""
        try:
            return await self.auth_manager.is_authenticated_async()
        except Exception as e:
            logger.error(f"Error checking authentication: {e}")
            return False
        
    def is_authenticated(self) -> bool:
        """Check if client is authenticated."""
        try:
            import asyncio
            
            # Check if we're in an async context and handle accordingly
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, use asyncio.to_thread
                async def async_check():
                    return await asyncio.to_thread(self.auth_manager.is_authenticated)
                
                # Run in thread to avoid blocking
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_check())
                    return future.result(timeout=5)
                    
            except RuntimeError:
                # No running loop, we can call directly
                return self.auth_manager.is_authenticated()
                
        except Exception as e:
            logger.error(f"Error checking authentication: {e}")
            return False
        
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
        
    async def get_current_playback_async(self) -> Optional[Dict[str, Any]]:
        """
        Get current playback information (async version).
        
        Returns:
            Dict with current track and playback state, or None
        """
        try:
            import asyncio
            spotify = await self._get_spotify_instance_async()
            if not spotify:
                return None
                
            # Use asyncio.to_thread to avoid blocking the event loop
            current = await asyncio.to_thread(spotify.current_playback)
            return current
        except Exception as e:
            logger.error(f"Error getting current playback: {e}")
            return None
            
    def get_current_playback(self) -> Optional[Dict[str, Any]]:
        """
        Get current playback information.
        
        Returns:
            Dict with current track and playback state, or None
        """
        try:
            import asyncio
            
            # Check if we're in an async context and handle accordingly
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, use the async version
                async def async_get():
                    return await self.get_current_playback_async()
                
                # Run in thread to avoid blocking
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_get())
                    return future.result(timeout=10)
                    
            except RuntimeError:
                # No running loop, we can call directly
                spotify = self._get_spotify_instance()
                if not spotify:
                    return None
                    
                current = spotify.current_playback()
                return current
        except Exception as e:
            logger.error(f"Error getting current playback: {e}")
            return None
            
    def play(self, uri: Optional[str] = None, context_uri: Optional[str] = None, device_id: Optional[str] = None) -> bool:
        """
        Start or resume playback.
        
        Args:
            uri: Optional Spotify URI to play (single track)
            context_uri: Optional Spotify context URI (album, playlist, artist)
            device_id: Optional device ID. Uses spotifyd device if not specified.
            
        Returns:
            bool: True if successful
        """
        try:
            spotify = self._get_spotify_instance()
            if not spotify:
                return False
                
            target_device = device_id or self.get_spotifyd_device_id()
            
            if context_uri:
                # Play a context (album, playlist, artist)
                spotify.start_playback(device_id=target_device, context_uri=context_uri)
            elif uri:
                # Play a single track
                spotify.start_playback(device_id=target_device, uris=[uri])
            else:
                # Resume playback
                spotify.start_playback(device_id=target_device)
            return True
        except Exception as e:
            logger.error(f"Error starting playback: {e}")
            return False
            
    def play_tracks(self, track_uris: List[str], device_id: Optional[str] = None) -> bool:
        """
        Play a list of tracks in sequence.
        
        Args:
            track_uris: List of Spotify track URIs to play
            device_id: Optional device ID. Uses spotifyd device if not specified.
            
        Returns:
            bool: True if successful
        """
        try:
            spotify = self._get_spotify_instance()
            if not spotify:
                return False
                
            target_device = device_id or self.get_spotifyd_device_id()
            
            # Play the list of tracks
            spotify.start_playback(device_id=target_device, uris=track_uris)
            return True
        except Exception as e:
            logger.error(f"Error playing track list: {e}")
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
            
    def search(self, query: str, types: List[str] = None, limit: int = 10, market: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for tracks, albums, artists, or playlists.
        
        Args:
            query: Search query
            types: List of types to search for ('track', 'album', 'artist', 'playlist')
            limit: Maximum number of results
            market: Market code for regional content (e.g., 'US', 'GB', 'SE')
            
        Returns:
            Dict with search results
        """
        try:
            spotify = self._get_spotify_instance()
            if not spotify:
                return {}
                
            search_types = types or ['track']
            
            # Build search parameters
            search_params = {
                'q': query,
                'limit': limit,
                'type': ','.join(search_types)
            }
            
            # Add market parameter if provided
            if market:
                search_params['market'] = market
                
            results = spotify.search(**search_params)
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {} 