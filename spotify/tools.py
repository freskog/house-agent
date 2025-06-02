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

# Global variable to track current music state across all tool calls
# This will be imported and used by the agent
current_music_state = {
    "current_song": "Nothing playing",
    "last_updated": None
}

def update_music_state(new_state: str):
    """Update the global music state."""
    global current_music_state
    import datetime
    current_music_state["current_song"] = new_state
    current_music_state["last_updated"] = datetime.datetime.now()
    
    # Also update the agent's current_music_info if it exists
    try:
        import agent
        if hasattr(agent, 'current_music_info'):
            agent.current_music_info["current_song"] = new_state
    except ImportError:
        pass  # Agent module not available

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
            if not current:
                result = "No music is currently playing"
                update_music_state(result)
                return result
            
            # Check if there's a track loaded (even if paused)
            if not current.get('item'):
                result = "No music is currently playing"
                update_music_state(result)
                return result
                
            track = current['item']
            artist = ', '.join([a['name'] for a in track['artists']])
            song = track['name']
            album = track['album']['name']
            is_playing = current.get('is_playing', False)
            
            result = f"Currently {'playing' if is_playing else 'paused'}: '{song}' by {artist} from '{album}'"
            update_music_state(result)
            return result
            
        except Exception as e:
            logger.error(f"Error getting current song: {e}")
            result = f"Error getting current song: {str(e)}"
            update_music_state("No music is currently playing")
            return result
            
    def play_music(query: Optional[str] = None, uri: Optional[str] = None) -> str:
        """Play music by search query or specific URI."""
        try:
            if uri:
                # Play specific URI
                success = spotify_client.play(uri=uri)
                if success:
                    # Update state to reflect music is now playing
                    # We could call get_current_song here, but let's just indicate something is playing
                    update_music_state("Music is playing")
                    return ""  # Return empty string for silent operation
                else:
                    return "Failed to start music playback"
                
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
                    state = f"Currently playing: '{song}' by {artist}"
                    update_music_state(state)
                    return ""  # Return empty string for silent operation
                else:
                    return "Failed to start music playback"
            else:
                # Resume playback
                success = spotify_client.play()
                if success:
                    update_music_state("Music resumed")
                    return ""  # Return empty string for silent operation
                else:
                    return "Failed to resume music playback"
                
        except Exception as e:
            logger.error(f"Error playing music: {e}")
            return f"Error playing music: {str(e)}"
            
    def pause_music(*args, **kwargs) -> str:
        """Pause the currently playing music."""
        try:
            success = spotify_client.pause()
            if success:
                update_music_state("Music paused")
                return ""  # Return empty string for silent operation
            else:
                return "Failed to pause music"
        except Exception as e:
            logger.error(f"Error pausing music: {e}")
            return f"Error pausing music: {str(e)}"
            
    def stop_music(*args, **kwargs) -> str:
        """Stop the currently playing music."""
        try:
            # Check if there's music to stop
            current = spotify_client.get_current_playback()
            if not current or not current.get('item'):
                result = "No music is currently playing"
                update_music_state(result)
                return result
            
            # Use pause to stop the music
            success = spotify_client.pause()
            if success:
                update_music_state("Music stopped")
                return ""  # Return empty string for silent operation
            else:
                return "Failed to stop music"
        except Exception as e:
            logger.error(f"Error stopping music: {e}")
            return f"Error stopping music: {str(e)}"
            
    def next_track(*args, **kwargs) -> str:
        """Skip to the next track."""
        try:
            success = spotify_client.next_track()
            return "" if success else "Failed to skip to next track"  # Silent on success
        except Exception as e:
            logger.error(f"Error skipping track: {e}")
            return f"Error skipping track: {str(e)}"
            
    def previous_track(*args, **kwargs) -> str:
        """Go back to the previous track."""
        try:
            success = spotify_client.previous_track()
            return "" if success else "Failed to go to previous track"  # Silent on success
        except Exception as e:
            logger.error(f"Error going to previous track: {e}")
            return f"Error going to previous track: {str(e)}"
            
    def set_volume(volume: int) -> str:
        """Set the music volume (0-100)."""
        try:
            success = spotify_client.set_volume(volume)
            return "" if success else "Failed to set volume"  # Silent on success
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
            name="stop_music",
            description="Stop the currently playing music",
            func=stop_music,
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