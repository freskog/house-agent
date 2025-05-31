#!/usr/bin/env python3
"""
Fixed Spotify OAuth test using correct spotipy methods.
"""

import os
import sys
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from urllib.parse import urlparse, parse_qs

def test_oauth_flow_fixed():
    """Test the Spotify OAuth authentication flow with correct method."""
    
    # Load environment variables
    load_dotenv()
    
    print("🎵 Spotify OAuth Test (Fixed)")
    print("=" * 50)
    
    # Check environment variables
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
    
    print(f"Client ID: {client_id[:10]}...{client_id[-5:] if client_id else 'NOT SET'}")
    print(f"Client Secret: {'SET' if client_secret else 'NOT SET'}")
    print(f"Redirect URI: {redirect_uri}")
    print()
    
    if not all([client_id, client_secret, redirect_uri]):
        print("❌ Missing required environment variables!")
        return False
    
    try:
        # Create auth manager directly
        print("1. Creating OAuth manager...")
        scopes = [
            "user-read-playback-state",
            "user-modify-playback-state", 
            "user-read-currently-playing",
            "playlist-read-private",
            "playlist-read-collaborative",
            "user-library-read",
            "streaming"
        ]
        
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=" ".join(scopes),
            cache_path=".spotify_cache",
            show_dialog=False,
            open_browser=False
        )
        
        # Check if already authenticated
        print("2. Checking existing authentication...")
        token_info = auth_manager.get_cached_token()
        
        if token_info and not auth_manager.is_token_expired(token_info):
            print("✅ Already authenticated!")
            
            # Test basic API call
            print("3. Testing API access...")
            spotify = spotipy.Spotify(auth=token_info['access_token'])
            devices = spotify.devices().get('devices', [])
            print(f"   Found {len(devices)} Spotify devices:")
            for device in devices:
                status = "🟢 Active" if device.get('is_active') else "⚪ Inactive"
                print(f"   • {device['name']} ({device['type']}) - {status}")
            
            return True
        
        else:
            print("🔐 Authentication required")
            print("3. Getting authorization URL...")
            auth_url = auth_manager.get_authorize_url()
            
            print(f"\n📱 Please visit this URL to authorize the app:")
            print(f"🔗 {auth_url}")
            print()
            print("After authorization, you'll be redirected to a page that says 'This site can't be reached'")
            print("Copy the ENTIRE URL from your browser's address bar and paste it here.")
            print()
            
            # Get response URL from user
            response_url = input("Paste the full redirect URL here: ").strip()
            
            if not response_url:
                print("❌ No URL provided")
                return False
            
            print("4. Extracting authorization code...")
            
            # Parse the response URL to get the code
            parsed_url = urlparse(response_url)
            params = parse_qs(parsed_url.query)
            
            if 'code' not in params:
                print("❌ No authorization code found in URL")
                print(f"   URL: {response_url}")
                return False
            
            auth_code = params['code'][0]
            print(f"   Authorization code: {auth_code[:20]}...")
            
            # Exchange authorization code for token
            print("5. Exchanging code for access token...")
            
            # Use get_access_token with the full response URL
            token_info = auth_manager.get_access_token(response_url)
            
            if token_info:
                print("✅ Authentication successful!")
                print(f"   Access token: {token_info['access_token'][:20]}...")
                
                # Test basic API call
                print("6. Testing API access...")
                spotify = spotipy.Spotify(auth=token_info['access_token'])
                devices = spotify.devices().get('devices', [])
                print(f"   Found {len(devices)} Spotify devices:")
                for device in devices:
                    status = "🟢 Active" if device.get('is_active') else "⚪ Inactive"
                    print(f"   • {device['name']} ({device['type']}) - {status}")
                
                return True
            else:
                print("❌ Failed to get access token")
                return False
                
    except Exception as e:
        print(f"❌ Error during OAuth flow: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_oauth_flow_fixed()
    print()
    if success:
        print("🎉 OAuth flow completed successfully!")
        print("The agent can now control Spotify playback.")
    else:
        print("❌ OAuth flow failed.")
        print("Please check your credentials and try again.")
    
    sys.exit(0 if success else 1) 