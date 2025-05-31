"""
LangChain tools for Spotify integration.
"""

from typing import Dict, Any, List, Optional
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
import json
import logging

from .client import SpotifyClient

logger = logging.getLogger(__name__)

class PlayMusicInput(BaseModel):
    """Input schema for playing music."""
    query: Optional[str] = Field(None, description="Search query for music to play (song, artist, album)")
    uri: Optional[str] = Field(None, description="Specific Spotify URI to play")

class SetVolumeInput(BaseModel):
    """Input schema for setting volume."""
    volume: int = Field(..., description="Volume level (0-100)", ge=0, le=100)

class SearchMusicInput(BaseModel):
    """Input schema for searching music."""
    query: str = Field(..., description="Search query for music")
    types: Optional[List[str]] = Field(["track"], description="Types to search for: track, album, artist, playlist")
    limit: Optional[int] = Field(10, description="Maximum number of results", ge=1, le=50)

def create_spotify_tools(spotify_client: SpotifyClient) -> List[Tool]:
    """
    Create LangChain tools for Spotify integration.
    
    Args:
        spotify_client: Authenticated Spotify client
        
    Returns:
        List of LangChain tools
    """
    
    def get_current_song(*args, **kwargs) -> str:
        """Get information about the currently playing song."""
        try:
            current = spotify_client.get_current_playback()
            if not current or not current.get('item'):
                return "No music is currently playing"
                
            track = current['item']
            artist = ', '.join([a['name'] for a in track['artists']])
            song = track['name']
            album = track['album']['name']
            is_playing = current.get('is_playing', False)
            
            return f"Currently {'playing' if is_playing else 'paused'}: '{song}' by {artist} from '{album}'"
            
        except Exception as e:
            logger.error(f"Error getting current song: {e}")
            return f"Error getting current song: {str(e)}"
            
    def play_music(query: Optional[str] = None, uri: Optional[str] = None) -> str:
        """Play music by search query or specific URI."""
        try:
            if uri:
                # Play specific URI
                success = spotify_client.play(uri=uri)
                return "Music started playing" if success else "Failed to start music playback"
                
            elif query:
                # Search for music and play first result
                results = spotify_client.search(query, types=['track'], limit=1)
                tracks = results.get('tracks', {}).get('items', [])
                
                if not tracks:
                    return f"No tracks found for query: {query}"
                    
                track = tracks[0]
                track_uri = track['uri']
                artist = ', '.join([a['name'] for a in track['artists']])
                song = track['name']
                
                success = spotify_client.play(uri=track_uri)
                if success:
                    return f"Now playing: '{song}' by {artist}"
                else:
                    return "Failed to start music playback"
            else:
                # Resume playback
                success = spotify_client.play()
                return "Music resumed" if success else "Failed to resume music playback"
                
        except Exception as e:
            logger.error(f"Error playing music: {e}")
            return f"Error playing music: {str(e)}"
            
    def pause_music(*args, **kwargs) -> str:
        """Pause the currently playing music."""
        try:
            success = spotify_client.pause()
            return "Music paused" if success else "Failed to pause music"
        except Exception as e:
            logger.error(f"Error pausing music: {e}")
            return f"Error pausing music: {str(e)}"
            
    def next_track(*args, **kwargs) -> str:
        """Skip to the next track."""
        try:
            success = spotify_client.next_track()
            return "Skipped to next track" if success else "Failed to skip to next track"
        except Exception as e:
            logger.error(f"Error skipping track: {e}")
            return f"Error skipping track: {str(e)}"
            
    def previous_track(*args, **kwargs) -> str:
        """Go back to the previous track."""
        try:
            success = spotify_client.previous_track()
            return "Went back to previous track" if success else "Failed to go to previous track"
        except Exception as e:
            logger.error(f"Error going to previous track: {e}")
            return f"Error going to previous track: {str(e)}"
            
    def set_volume(volume: int) -> str:
        """Set the music volume (0-100)."""
        try:
            success = spotify_client.set_volume(volume)
            return f"Volume set to {volume}%" if success else "Failed to set volume"
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
            return f"Error setting volume: {str(e)}"
            
    def search_music(query: str, types: Optional[List[str]] = None, limit: Optional[int] = 10) -> str:
        """Search for music and return results."""
        try:
            search_types = types or ['track']
            
            results = spotify_client.search(query, types=search_types, limit=limit)
            
            output = []
            for search_type in search_types:
                items = results.get(f"{search_type}s", {}).get("items", [])
                if items:
                    output.append(f"\n{search_type.title()}s:")
                    for i, item in enumerate(items, 1):
                        if search_type == 'track':
                            artist = ', '.join([a['name'] for a in item['artists']])
                            output.append(f"  {i}. {item['name']} by {artist} (URI: {item['uri']})")
                        elif search_type == 'album':
                            artist = ', '.join([a['name'] for a in item['artists']])
                            output.append(f"  {i}. {item['name']} by {artist} (URI: {item['uri']})")
                        elif search_type == 'artist':
                            output.append(f"  {i}. {item['name']} (URI: {item['uri']})")
                        elif search_type == 'playlist':
                            output.append(f"  {i}. {item['name']} by {item['owner']['display_name']} (URI: {item['uri']})")
                            
            return '\n'.join(output) if output else f"No results found for: {query}"
            
        except Exception as e:
            logger.error(f"Error searching music: {e}")
            return f"Error searching music: {str(e)}"
            
    def get_devices(*args, **kwargs) -> str:
        """Get available Spotify devices."""
        try:
            devices = spotify_client.get_devices()
            if not devices:
                return "No Spotify devices found"
                
            output = ["Available Spotify devices:"]
            for device in devices:
                status = "🟢 Active" if device.get('is_active') else "⚪ Inactive"
                output.append(f"  • {device['name']} ({device['type']}) - {status}")
                
            return '\n'.join(output)
        except Exception as e:
            logger.error(f"Error getting devices: {e}")
            return f"Error getting devices: {str(e)}"
    
    # Create the tools
    tools = [
        Tool(
            name="get_current_song",
            description="Get information about the currently playing song",
            func=get_current_song,
            return_direct=False
        ),
        StructuredTool(
            name="play_music", 
            description="Play music by search query or resume playback. Use query to search for music or uri for specific track",
            func=play_music,
            args_schema=PlayMusicInput
        ),
        Tool(
            name="pause_music",
            description="Pause the currently playing music",
            func=pause_music,
            return_direct=False
        ),
        Tool(
            name="next_track",
            description="Skip to the next track",
            func=next_track,
            return_direct=False
        ),
        Tool(
            name="previous_track",
            description="Go back to the previous track",
            func=previous_track,
            return_direct=False
        ),
        StructuredTool(
            name="set_volume",
            description="Set the music volume (0-100)",
            func=set_volume,
            args_schema=SetVolumeInput
        ),
        StructuredTool(
            name="search_music",
            description="Search for music (tracks, albums, artists, playlists) and get URIs for playback",
            func=search_music,
            args_schema=SearchMusicInput
        ),
        Tool(
            name="get_spotify_devices",
            description="Get list of available Spotify devices",
            func=get_devices,
            return_direct=False
        )
    ]
    
    return tools 