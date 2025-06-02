"""
LangChain tools for Spotify integration.
"""

from typing import Dict, Any, List, Optional
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
import json
import logging
import random
import re

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
    query: Optional[str] = Field(None, description="Search query for music to play (song, artist, album, genre)")
    uri: Optional[str] = Field(None, description="Specific Spotify URI to play")

class SetVolumeInput(BaseModel):
    """Input schema for setting volume."""
    volume: int = Field(..., description="Volume level (0-100)", ge=0, le=100)

class SearchMusicInput(BaseModel):
    """Input schema for searching music."""
    query: str = Field(..., description="Search query for music")
    types: Optional[List[str]] = Field(["track"], description="Types to search for: track, album, artist, playlist")
    limit: Optional[int] = Field(10, description="Maximum number of results", ge=1, le=50)

class CreateRadioInput(BaseModel):
    """Input schema for creating radio stations."""
    seed_artists: Optional[List[str]] = Field(None, description="List of artist names to use as seeds")
    seed_tracks: Optional[List[str]] = Field(None, description="List of track names to use as seeds") 
    seed_genres: Optional[List[str]] = Field(None, description="List of genre names to use as seeds")
    limit: Optional[int] = Field(20, description="Number of tracks to include in radio station", ge=1, le=50)

def _is_genre_or_style_query(query: str) -> bool:
    """Check if a query is asking for a genre or musical style."""
    genre_keywords = [
        'rock', 'pop', 'jazz', 'blues', 'country', 'hip hop', 'rap', 'electronic', 
        'classical', 'reggae', 'funk', 'soul', 'r&b', 'folk', 'metal', 'punk',
        'indie', 'alternative', 'dance', 'house', 'techno', 'ambient', 'chill',
        'soft', 'mellow', 'upbeat', 'relaxing', 'energetic', 'workout', 'party'
    ]
    
    era_keywords = [
        '60s', '70s', '80s', '90s', '2000s', 'sixties', 'seventies', 'eighties', 'nineties'
    ]
    
    style_keywords = [
        'ballad', 'anthem', 'instrumental', 'acoustic', 'live', 'remix'
    ]
    
    query_lower = query.lower()
    
    # Check for genre/style/era keywords
    for keyword in genre_keywords + era_keywords + style_keywords:
        if keyword in query_lower:
            return True
    
    # Check for patterns like "something music" or "something songs"
    if re.search(r'\b(music|songs|tracks|playlist)\b', query_lower):
        return True
    
    return False

def _get_recommendations_from_track(spotify_client: SpotifyClient, track_uri: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get track recommendations based on a seed track."""
    try:
        # Extract track ID from URI
        track_id = track_uri.split(':')[-1]
        
        # Get audio features for the track to use as seeds
        features = spotify_client._get_spotify_instance().audio_features([track_id])
        if not features or not features[0]:
            logger.warning(f"No audio features found for track {track_id}")
            return []
        
        audio_features = features[0]
        
        # Build recommendation parameters based on audio features
        params = {
            'seed_tracks': [track_id],
            'limit': limit,
            'target_valence': audio_features.get('valence', 0.5),
            'target_energy': audio_features.get('energy', 0.5),
            'target_danceability': audio_features.get('danceability', 0.5)
        }
        
        # Get recommendations
        recommendations = spotify_client._get_spotify_instance().recommendations(**params)
        return recommendations.get('tracks', [])
        
    except Exception as e:
        logger.error(f"Error getting recommendations for track {track_uri}: {e}")
        return []

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
        """Play music by search query or specific URI with smart context selection."""
        try:
            if uri:
                # Play specific URI - check if it's a track or context
                if uri.startswith('spotify:track:'):
                    # For individual tracks, try to get recommendations to create a better experience
                    recommendations = _get_recommendations_from_track(spotify_client, uri, 49)
                    if recommendations:
                        # Create a playlist with the original track + recommendations
                        track_uris = [uri] + [track['uri'] for track in recommendations]
                        success = spotify_client.play_tracks(track_uris)
                    else:
                        # Fall back to playing just the track
                        success = spotify_client.play(uri=uri)
                else:
                    # It's a context URI (album, playlist, artist) - play directly
                    success = spotify_client.play(context_uri=uri)
                
                if success:
                    update_music_state("Music is playing")
                    return ""
                else:
                    return "Failed to start music playback"
                
            elif query:
                # Determine if this is a genre/style query or specific track/artist search
                is_genre_query = _is_genre_or_style_query(query)
                
                if is_genre_query:
                    # For genre queries, search for playlists first for better continuous playback
                    results = spotify_client.search(query, types=['playlist', 'album'], limit=10)
                    
                    # Try playlists first
                    playlists = results.get('playlists', {}).get('items', [])
                    if playlists:
                        # Randomize playlist selection
                        playlist = random.choice(playlists)
                        playlist_uri = playlist['uri']
                        
                        success = spotify_client.play(context_uri=playlist_uri)
                        if success:
                            state = f"Playing playlist: '{playlist['name']}' by {playlist['owner']['display_name']}"
                            update_music_state(state)
                            return ""
                    
                    # If no playlists, try albums
                    albums = results.get('albums', {}).get('items', [])
                    if albums:
                        album = random.choice(albums)
                        album_uri = album['uri']
                        
                        success = spotify_client.play(context_uri=album_uri)
                        if success:
                            artist = ', '.join([a['name'] for a in album['artists']])
                            state = f"Playing album: '{album['name']}' by {artist}"
                            update_music_state(state)
                            return ""
                    
                    # Fall back to track search with recommendations
                    track_results = spotify_client.search(query, types=['track'], limit=20)
                    tracks = track_results.get('tracks', {}).get('items', [])
                    if tracks:
                        # Randomize track selection
                        selected_track = random.choice(tracks)
                        track_uri = selected_track['uri']
                        
                        # Get recommendations based on this track
                        recommendations = _get_recommendations_from_track(spotify_client, track_uri, 49)
                        if recommendations:
                            track_uris = [track_uri] + [track['uri'] for track in recommendations]
                            success = spotify_client.play_tracks(track_uris)
                        else:
                            success = spotify_client.play(uri=track_uri)
                        
                        if success:
                            artist = ', '.join([a['name'] for a in selected_track['artists']])
                            state = f"Playing {query} starting with: '{selected_track['name']}' by {artist}"
                            update_music_state(state)
                            return ""
                    
                    return f"No music found for: {query}"
                    
                else:
                    # Specific track/artist search - search tracks first, then try albums/artists
                    results = spotify_client.search(query, types=['track', 'album', 'artist'], limit=5)
                    
                    # Try tracks first
                    tracks = results.get('tracks', {}).get('items', [])
                    if tracks:
                        # For specific searches, use the first result but get recommendations
                        track = tracks[0]
                        track_uri = track['uri']
                        
                        # Get recommendations for continuous playback
                        recommendations = _get_recommendations_from_track(spotify_client, track_uri, 49)
                        if recommendations:
                            track_uris = [track_uri] + [track['uri'] for track in recommendations]
                            success = spotify_client.play_tracks(track_uris)
                        else:
                            success = spotify_client.play(uri=track_uri)
                        
                        if success:
                            artist = ', '.join([a['name'] for a in track['artists']])
                            state = f"Playing: '{track['name']}' by {artist} (with similar songs)"
                            update_music_state(state)
                            return ""
                    
                    # Try albums
                    albums = results.get('albums', {}).get('items', [])
                    if albums:
                        album = albums[0]  # Use first result for specific searches
                        album_uri = album['uri']
                        
                        success = spotify_client.play(context_uri=album_uri)
                        if success:
                            artist = ', '.join([a['name'] for a in album['artists']])
                            state = f"Playing album: '{album['name']}' by {artist}"
                            update_music_state(state)
                            return ""
                    
                    # Try artists
                    artists = results.get('artists', {}).get('items', [])
                    if artists:
                        artist = artists[0]
                        # Get artist's top tracks for playback
                        top_tracks = spotify_client._get_spotify_instance().artist_top_tracks(artist['id'])
                        if top_tracks and top_tracks.get('tracks'):
                            track_uris = [track['uri'] for track in top_tracks['tracks'][:20]]
                            success = spotify_client.play_tracks(track_uris)
                            if success:
                                state = f"Playing top songs by {artist['name']}"
                                update_music_state(state)
                                return ""
                    
                    return f"No tracks found for query: {query}"
            else:
                # Resume playback
                success = spotify_client.play()
                if success:
                    update_music_state("Music resumed")
                    return ""
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
    
    def create_radio_station(
        seed_artists: Optional[List[str]] = None, 
        seed_tracks: Optional[List[str]] = None, 
        seed_genres: Optional[List[str]] = None, 
        limit: Optional[int] = 20
    ) -> str:
        """Create a radio station based on seed artists, tracks, or genres."""
        try:
            spotify = spotify_client._get_spotify_instance()
            if not spotify:
                return "Failed to connect to Spotify"
            
            # Convert artist/track names to IDs if provided
            artist_ids = []
            track_ids = []
            
            if seed_artists:
                for artist_name in seed_artists[:2]:  # Limit to 2 artists
                    results = spotify.search(artist_name, type='artist', limit=1)
                    if results['artists']['items']:
                        artist_ids.append(results['artists']['items'][0]['id'])
            
            if seed_tracks:
                for track_name in seed_tracks[:2]:  # Limit to 2 tracks
                    results = spotify.search(track_name, type='track', limit=1)
                    if results['tracks']['items']:
                        track_ids.append(results['tracks']['items'][0]['id'])
            
            # Validate genres if provided
            valid_genres = seed_genres[:2] if seed_genres else []  # Limit to 2 genres
            
            # Need at least one seed
            if not artist_ids and not track_ids and not valid_genres:
                return "Please provide at least one seed artist, track, or genre"
            
            # Get recommendations
            recommendations = spotify.recommendations(
                seed_artists=artist_ids[:2],
                seed_tracks=track_ids[:2], 
                seed_genres=valid_genres[:2],
                limit=limit
            )
            
            if not recommendations or not recommendations.get('tracks'):
                return "No recommendations found"
                
            # Play the recommended tracks
            track_uris = [track['uri'] for track in recommendations['tracks']]
            success = spotify_client.play_tracks(track_uris)
            
            if success:
                seed_info = []
                if seed_artists:
                    seed_info.append(f"artists: {', '.join(seed_artists[:2])}")
                if seed_tracks:
                    seed_info.append(f"tracks: {', '.join(seed_tracks[:2])}")
                if seed_genres:
                    seed_info.append(f"genres: {', '.join(seed_genres[:2])}")
                
                state = f"Playing radio station based on {', '.join(seed_info)}"
                update_music_state(state)
                return ""
            else:
                return "Failed to start radio station playback"
                
        except Exception as e:
            logger.error(f"Error creating radio station: {e}")
            return f"Error creating radio station: {str(e)}"
    
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
            description="Play music by search query with smart context selection. Supports specific tracks, artists, albums, genres, and musical styles. Automatically creates continuous playback experience.",
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
        ),
        StructuredTool(
            name="create_radio_station",
            description="Create a personalized radio station based on seed artists, tracks, or genres. Perfect for discovering new music similar to what you like.",
            func=create_radio_station,
            args_schema=CreateRadioInput
        ),
    ]
    
    return tools 