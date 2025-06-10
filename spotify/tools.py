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
import concurrent.futures
import asyncio

from .client import SpotifyClient

logger = logging.getLogger(__name__)

# Global variable to track current music state across all tool calls
# This will be imported and used by the agent
current_music_state = {
    "current_song": "Nothing playing",
    "last_updated": None
}

# Lazy initialization - don't create client until needed
_spotify_client = None

def _get_spotify_client():
    """Get or create the Spotify client lazily."""
    global _spotify_client
    if _spotify_client is None:
        from .client import SpotifyClient
        _spotify_client = SpotifyClient()
    return _spotify_client

def update_music_state(new_state: str):
    """Update the global music state."""
    global current_music_state
    import datetime
    current_music_state["current_song"] = new_state
    current_music_state["last_updated"] = datetime.datetime.now()
    
    # Legacy agent global removed - state now managed by nodes

class PlayMusicInput(BaseModel):
    """Input schema for playing music."""
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="""REQUIRED context object with type and market. Must be provided for every call!
        
        Format: {"type": "album|artist|track|genre", "market": "country_code"}
        
        Examples:
        - For Swedish artist: {"type": "artist", "market": "SE"}
        - For Spanish song: {"type": "track", "market": "ES"}
        - For genre/mood: {"type": "genre", "market": "IE"}
        - For Swedish album: {"type": "album", "market": "SE"}
        
        Type guidelines:
        - Use "genre" for style/mood requests (rock, pop, chill, etc.)
        - Use "artist" for specific artists (ABBA, Miss Li, etc.)
        - Use "track" for specific songs
        - Use "album" for album requests
        
        Market codes: SE (Swedish), ES (Spanish), DE (German), FR (French), IE (Ireland/English)
        
        THIS FIELD IS MANDATORY - ALWAYS provide it!"""
    )
    query: Optional[str] = Field(None, description="Search query for music to play (song, artist, album, genre)")
    uri: Optional[str] = Field(None, description="Specific Spotify URI to play")
    market: Optional[str] = Field("IE", description="Market code for regional content (e.g., 'SE' for Sweden, 'ES' for Spain)")
    create_playlist: Optional[bool] = Field(True, description="Whether to create a playlist from the results")

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

# Global async function for startup use
async def get_current_song_startup_async() -> str:
    """Get information about the currently playing song for startup (async version)."""
    try:
        spotify_client = _get_spotify_client()
        current = await spotify_client.get_current_playback_async()
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

def create_spotify_tools() -> List[Tool]:
    """
    Create LangChain tools for Spotify integration with lazy client initialization.
    
    Returns:
        List of LangChain tools
    """
    
    # Helper functions that use lazy initialization
    def _get_recommendations_from_track(track_uri: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get track recommendations based on a seed track."""
        try:
            spotify_client = _get_spotify_client()
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

    def _get_query_context(query: str) -> Dict[str, Any]:
        """Get context information about the query.
        This provides basic fallback context detection if the LLM doesn't provide context."""
        query_lower = query.lower()
        
        # Language/market detection
        market = "IE"  # Default to Ireland
        if any(word in query_lower for word in ["swedish", "sverige", "sweden", "miss li", "robyn", "roxette", "abba", "ace of base"]):
            market = "SE"
        elif any(word in query_lower for word in ["spanish", "español", "spain", "españa"]):
            market = "ES"
        elif any(word in query_lower for word in ["german", "deutsch", "germany", "deutschland"]):
            market = "DE"
        elif any(word in query_lower for word in ["french", "français", "france"]):
            market = "FR"
        elif any(word in query_lower for word in ["italian", "italiano", "italy", "italia"]):
            market = "IT"
        elif any(word in query_lower for word in ["norwegian", "norsk", "norway", "norge"]):
            market = "NO"
        elif any(word in query_lower for word in ["danish", "dansk", "denmark", "danmark"]):
            market = "DK"
        
        # Context type detection - prioritize specific requests
        context_type = "genre"  # Default to genre for better variety
        
        # Check for album indicators first (most specific)
        if any(word in query_lower for word in ["album", "full album", "entire album", "latest album", "new album", "senaste album"]):
            context_type = "album"
        # Check for specific artist indicators
        elif any(word in query_lower for word in ["by ", "from ", "artist"]):
            context_type = "artist"
        # Check for specific track indicators
        elif any(word in query_lower for word in ["song", "track", "play the song"]):
            context_type = "track"
        # Check for playlist indicators
        elif any(word in query_lower for word in ["playlist"]):
            context_type = "playlist"
        # If we have a specific artist name, treat as artist search
        elif any(artist in query_lower for artist in ["miss li", "robyn", "roxette", "abba", "ace of base"]):
            context_type = "artist"
        # Keep genre as default for style/mood requests
        
        return {
            "type": context_type,
            "market": market
        }

    def _create_playlist_from_tracks(tracks: List[Dict[str, Any]], name: str, description: str) -> Optional[str]:
        """Create a playlist from a list of tracks."""
        try:
            spotify = _get_spotify_client()._get_spotify_instance()
            if not spotify:
                return None
                
            # Get current user
            user = spotify.current_user()
            if not user:
                return None
                
            # Create playlist
            playlist = spotify.user_playlist_create(
                user=user['id'],
                name=name,
                description=description,
                public=False
            )
            
            if not playlist:
                return None
                
            # Add tracks to playlist
            track_uris = [track['uri'] for track in tracks]
            spotify.playlist_add_items(playlist['id'], track_uris)
            
            return playlist['uri']
        except Exception as e:
            logger.error(f"Error creating playlist: {e}")
            return None

    def _play_track_with_recommendations(track_uri: str, create_playlist: bool = True) -> tuple[bool, str]:
        """Play a track with recommendations."""
        recommendations = _get_recommendations_from_track(track_uri, 29)  # Get 29 recommendations + 1 original = 30 total
        if not recommendations:
            return _get_spotify_client().play(uri=track_uri), ""

        if create_playlist:
            playlist_uri = _create_playlist_from_tracks(
                [{"uri": track_uri}] + recommendations,
                "Recommended Tracks",
                "Playlist created from your selected track and similar recommendations"
            )
            if playlist_uri:
                return _get_spotify_client().play(context_uri=playlist_uri), ""
        
        track_uris = [track_uri] + [track['uri'] for track in recommendations]
        return _get_spotify_client().play_tracks(track_uris), ""

    def _play_search_results(results: Dict[str, Any], context: Dict[str, Any], create_playlist: bool = True) -> tuple[bool, str]:
        """Play music from search results based on context, using radio station approach like mobile app."""
        query_type = context.get("type", "genre")
        
        # For artist requests - create radio station like mobile app
        if query_type == "artist":
            artists = results.get('artists', {}).get('items', [])
            if artists:
                artist = artists[0]
                # Use create_radio_station function for true radio experience
                try:
                    radio_result = create_radio_station(seed_artists=[artist['name']], limit=30)
                    if not radio_result:  # Empty result means success
                        return True, f"Playing radio station based on {artist['name']} (30 songs with similar artists)"
                    else:
                        # Fallback to top tracks if radio creation fails
                        top_tracks = _get_spotify_client()._get_spotify_instance().artist_top_tracks(artist['id'], country=context.get("market", "IE"))
                        if top_tracks and top_tracks.get('tracks'):
                            track_uris = [track['uri'] for track in top_tracks['tracks'][:20]]
                            success = _get_spotify_client().play_tracks(track_uris)
                            if success:
                                return True, f"Playing top songs by {artist['name']} (20 tracks)"
                except Exception as e:
                    logger.error(f"Radio station creation failed for artist {artist['name']}: {e}")
                    # Fallback to current method
                    pass
        
        # For genre requests - create radio station with genre seeds
        if query_type == "genre":
            # Extract genre from query for radio station
            query_text = results.get('_query', '')  # We'll pass this in the search
            if query_text:
                try:
                    # Try to create a genre-based radio station
                    radio_result = create_radio_station(seed_genres=[query_text], limit=30)
                    if not radio_result:  # Empty result means success
                        return True, f"Playing {query_text} radio station (30 songs)"
                except Exception as e:
                    logger.error(f"Genre radio station failed for {query_text}: {e}")
            
            # Fallback to current genre handling with multiple sources
            all_tracks = []
            
            # Get tracks from search results
            tracks = results.get('tracks', {}).get('items', [])
            if tracks:
                all_tracks.extend(tracks[:10])  # Take top 10 tracks
            
            # Get tracks from albums
            albums = results.get('albums', {}).get('items', [])
            for album in albums[:3]:  # Take a few tracks from top albums
                album_tracks = _get_spotify_client()._get_spotify_instance().album_tracks(album['id'], limit=5)
                if album_tracks and album_tracks.get('items'):
                    all_tracks.extend(album_tracks['items'][:3])  # Take 3 tracks per album
            
            # Get tracks from artists
            artists = results.get('artists', {}).get('items', [])
            for artist in artists[:3]:  # Take tracks from top artists
                top_tracks = _get_spotify_client()._get_spotify_instance().artist_top_tracks(artist['id'], country=context.get("market", "IE"))
                if top_tracks and top_tracks.get('tracks'):
                    all_tracks.extend(top_tracks['tracks'][:3])  # Take 3 tracks per artist
            
            # If we have tracks, create a varied playlist
            if all_tracks:
                # Shuffle for variety and limit to reasonable size
                random.shuffle(all_tracks)
                selected_tracks = all_tracks[:25]  # More tracks for radio-like experience
                
                if create_playlist:
                    playlist_uri = _create_playlist_from_tracks(
                        selected_tracks,
                        f"{query_text.title()} Mix" if query_text else "Genre Mix",
                        f"Radio-style playlist for {query_text}" if query_text else "Mixed playlist for your music request"
                    )
                    if playlist_uri:
                        success = _get_spotify_client().play(context_uri=playlist_uri)
                        if success:
                            return True, f"Playing {query_text} mix with {len(selected_tracks)} songs"
                
                # Fallback to track list if playlist creation fails
                track_uris = [track['uri'] for track in selected_tracks if track.get('uri')]
                if track_uris:
                    success = _get_spotify_client().play_tracks(track_uris)
                    if success:
                        return True, f"Playing {len(track_uris)} songs for your request"
        
        # For track requests - create radio station based on that track
        if query_type == "track":
            tracks = results.get('tracks', {}).get('items', [])
            if tracks:
                track = tracks[0]
                # Try to create radio station based on this track
                try:
                    radio_result = create_radio_station(seed_tracks=[track['name']], limit=30)
                    if not radio_result:  # Empty result means success
                        artist = ', '.join([a['name'] for a in track['artists']])
                        return True, f"Playing radio station starting with '{track['name']}' by {artist} (30 similar songs)"
                except Exception as e:
                    logger.error(f"Track radio station failed: {e}")
                
                # Fallback to track with recommendations
                success, _ = _play_track_with_recommendations(track['uri'], create_playlist)
                if success:
                    artist = ', '.join([a['name'] for a in track['artists']])
                    return True, f"Playing: '{track['name']}' by {artist} (with similar songs)"
        
        # Try playlists first if that's what was requested
        if query_type == "playlist":
            playlists = results.get('playlists', {}).get('items', [])
            if playlists:
                playlist = playlists[0]
                success = _get_spotify_client().play(context_uri=playlist['uri'])
                if success:
                    return True, f"Playing playlist: '{playlist['name']}' by {playlist['owner']['display_name']}"
        
        # Try albums (full album context - keep this as is since albums should play fully)
        albums = results.get('albums', {}).get('items', [])
        if albums:
            album = albums[0]
            success = _get_spotify_client().play(context_uri=album['uri'])
            if success:
                artist = ', '.join([a['name'] for a in album['artists']])
                return True, f"Playing album: '{album['name']}' by {artist}"
        
        # Final fallback - try any tracks available
        tracks = results.get('tracks', {}).get('items', [])
        if tracks:
            track = tracks[0]
            success, _ = _play_track_with_recommendations(track['uri'], create_playlist)
            if success:
                artist = ', '.join([a['name'] for a in track['artists']])
                return True, f"Playing: '{track['name']}' by {artist} (with similar songs)"
        
        # Final fallback - try any artists available
        artists = results.get('artists', {}).get('items', [])
        if artists:
            artist = artists[0]
            top_tracks = _get_spotify_client()._get_spotify_instance().artist_top_tracks(artist['id'], country=context.get("market", "IE"))
            if top_tracks and top_tracks.get('tracks'):
                track_uris = [track['uri'] for track in top_tracks['tracks'][:20]]
                success = _get_spotify_client().play_tracks(track_uris)
                if success:
                    return True, f"Playing top songs by {artist['name']}"
        
        return False, ""

    def play_music(context: Dict[str, Any] = None, query: Optional[str] = None, uri: Optional[str] = None, market: str = "IE", create_playlist: bool = True) -> str:
        """Play music by search query or specific URI with smart context selection."""
        try:
            # Critical validation - provide immediate feedback for missing context
            if context is None or not isinstance(context, dict) or 'type' not in context or 'market' not in context:
                # Return a structured error that LangChain/LangGraph will pass back to the AI
                error_msg = """VALIDATION ERROR: Missing required 'context' field.

The play_music tool requires a context parameter in this exact format:
{"context": {"type": "artist|track|album|genre", "market": "country_code"}}

For Swedish artist 'Bolaget', the correct call is:
{"context": {"type": "artist", "market": "SE"}, "query": "Bolaget"}

Please retry with the correct context parameter."""
                
                return error_msg
            
            if uri:
                # Play specific URI
                if uri.startswith('spotify:track:'):
                    success, _ = _play_track_with_recommendations(uri, create_playlist)
                else:
                    success = _get_spotify_client().play(context_uri=uri)
                
                if success:
                    update_music_state("Music is playing")
                    return ""
                return "Failed to start music playback"
                
            elif query:
                # Context is now required from the LLM, but provide fallback just in case
                if not context or not isinstance(context, dict):
                    logger.warning(f"Invalid context provided for query '{query}': {context}, using fallback detection")
                    context = _get_query_context(query)
                
                # Search based on provided context
                search_types = []
                query_type = context.get("type", "genre")
                
                if query_type == "genre":
                    # For genre searches, search across all types for variety
                    search_types = ["track", "album", "artist", "playlist"]
                elif query_type == "artist":
                    search_types = ["artist", "track", "album"]
                elif query_type == "album":
                    search_types = ["album", "track"]
                elif query_type == "track":
                    search_types = ["track", "artist"]
                elif query_type == "playlist":
                    search_types = ["playlist", "track"]
                else:
                    # Default fallback
                    search_types = ["track", "album", "artist"]
                
                logger.info(f"🎵 Playing music with context: {context}, search_types: {search_types}")
                
                results = _get_spotify_client().search(
                    query, 
                    types=search_types, 
                    limit=20, 
                    market=context.get("market", market)
                )
                
                # Add query text to results for genre radio station creation
                results['_query'] = query
                
                success, state = _play_search_results(results, context, create_playlist)
                if success:
                    update_music_state(state)
                    return ""
                return f"No tracks found for query: {query}"
                
            else:
                # Resume playback
                success = _get_spotify_client().play()
                if success:
                    update_music_state("Music resumed")
                    return ""
                return "Failed to resume music playback"
                
        except Exception as e:
            logger.error(f"Error playing music: {e}")
            return f"Error playing music: {str(e)}"

    # Tool functions
    async def get_current_song_async(*args, **kwargs) -> str:
        """Get information about the currently playing song (async version)."""
        try:
            current = _get_spotify_client().get_current_playback_async()
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
            
    def get_current_song(*args, **kwargs) -> str:
        """Get information about the currently playing song."""
        try:
            # LangChain tools must be synchronous - there's no way around this
            # The blocking I/O warnings from ASGI servers are due to spotipy's OAuth file operations
            # which we can't easily make async without rewriting the entire library
            current = _get_spotify_client().get_current_playback()
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
    
    def pause_music(*args, **kwargs) -> str:
        """Pause the currently playing music."""
        try:
            success = _get_spotify_client().pause()
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
            # Check current playback state first
            current = _get_spotify_client().get_current_playback()
            
            # If no music session at all, consider it already stopped
            if not current or not current.get('item'):
                update_music_state("Music stopped")
                return ""  # Silent success - already stopped
            
            # If music is already paused, consider it already stopped
            if not current.get('is_playing', False):
                update_music_state("Music stopped")
                return ""  # Silent success - already paused/stopped
            
            # Music is playing, so pause it
            success = _get_spotify_client().pause()
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
            success = _get_spotify_client().next_track()
            return "" if success else "Failed to skip to next track"  # Silent on success
        except Exception as e:
            logger.error(f"Error skipping track: {e}")
            return f"Error skipping track: {str(e)}"
            
    def previous_track(*args, **kwargs) -> str:
        """Go back to the previous track."""
        try:
            success = _get_spotify_client().previous_track()
            return "" if success else "Failed to go to previous track"  # Silent on success
        except Exception as e:
            logger.error(f"Error going to previous track: {e}")
            return f"Error going to previous track: {str(e)}"
            
    def set_volume(volume: int) -> str:
        """Set the music volume (0-100)."""
        try:
            success = _get_spotify_client().set_volume(volume)
            return "" if success else "Failed to set volume"  # Silent on success
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
            return f"Error setting volume: {str(e)}"
            
    def search_music(query: str, types: Optional[List[str]] = None, limit: Optional[int] = 10) -> str:
        """Search for music and return results."""
        try:
            search_types = types or ['track']
            
            results = _get_spotify_client().search(query, types=search_types, limit=limit)
            
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
            devices = _get_spotify_client().get_devices()
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
            spotify = _get_spotify_client()._get_spotify_instance()
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
            success = _get_spotify_client().play_tracks(track_uris)
            
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
            description="""Play music with radio station experience like Spotify mobile app. 

            🚨 VALIDATION ENFORCED: This tool validates the 'context' field and will return an error with instructions if missing! 🚨
            
            The context field must be provided as:
            {"context": {"type": "artist|track|album|genre", "market": "country_code"}}
            
            Radio Station Behavior (like mobile app):
            - Artist requests: Creates 30-song radio station based on the artist + similar artists
            - Track requests: Creates 30-song radio station starting with that track + similar songs  
            - Genre requests: Creates radio station with 30 songs in that genre/style
            - Album requests: Plays the full album (traditional album experience)
            
            Examples of CORRECT calls:
            - Swedish artist: {"context": {"type": "artist", "market": "SE"}, "query": "ABBA"}
            - Spanish song: {"context": {"type": "track", "market": "ES"}, "query": "Despacito"}
            - Genre request: {"context": {"type": "genre", "market": "IE"}, "query": "rock"}
            
            If you call without context, you will receive a detailed error message with instructions.""",
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