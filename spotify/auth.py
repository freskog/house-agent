"""
OAuth authentication handler for Spotify Web API.
"""

import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import Optional
import logging
from urllib.parse import urlparse, parse_qs

from .config import SpotifyConfig

logger = logging.getLogger(__name__)

class SpotifyAuthManager:
    """Manages Spotify OAuth authentication and token refresh."""
    
    def __init__(self, config: SpotifyConfig):
        """
        Initialize the auth manager.
        
        Args:
            config: Spotify configuration containing credentials
        """
        self.config = config
        self.auth_manager = SpotifyOAuth(
            client_id=config.client_id,
            client_secret=config.client_secret,
            redirect_uri=config.redirect_uri,
            scope=" ".join(config.scopes),
            cache_path=config.cache_path,
            show_dialog=False,
            open_browser=False  # We'll handle this manually
        )
        
    def get_auth_url(self) -> str:
        """
        Get the authorization URL for user to visit.
        
        Returns:
            str: Authorization URL
        """
        return self.auth_manager.get_authorize_url()
        
    def get_access_token(self, authorization_response_url: Optional[str] = None) -> Optional[str]:
        """
        Get access token from cached token or authorization response.
        
        Args:
            authorization_response_url: The full URL that Spotify redirected to
            
        Returns:
            str: Access token if available, None otherwise
        """
        try:
            if authorization_response_url:
                # Extract authorization code from response URL and get token
                parsed_url = urlparse(authorization_response_url)
                params = parse_qs(parsed_url.query)
                
                if 'code' in params:
                    # Use the working method: pass just the code
                    auth_code = params['code'][0]
                    token_info = self.auth_manager.get_access_token(code=auth_code, as_dict=True)
                else:
                    # Fallback to original method
                    token_info = self.auth_manager.get_access_token(authorization_response_url, as_dict=True)
            else:
                # Try to get cached token or refresh if needed
                token_info = self.auth_manager.get_cached_token()
                
                # If no cached token, we need user authorization
                if not token_info:
                    logger.warning("No cached token found. User authorization required.")
                    return None
                    
                # If token is expired, try to refresh
                if self.auth_manager.is_token_expired(token_info):
                    logger.info("Token expired, attempting refresh...")
                    token_info = self.auth_manager.refresh_access_token(token_info['refresh_token'])
                    
            return token_info['access_token'] if token_info else None
            
        except Exception as e:
            logger.error(f"Error getting access token: {e}")
            return None
            
    def is_authenticated(self) -> bool:
        """
        Check if we have a valid access token.
        
        Returns:
            bool: True if authenticated, False otherwise
        """
        token = self.get_access_token()
        return token is not None
        
    def clear_cache(self):
        """Clear the token cache file."""
        try:
            if os.path.exists(self.config.cache_path):
                os.remove(self.config.cache_path)
                logger.info("Token cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}") 