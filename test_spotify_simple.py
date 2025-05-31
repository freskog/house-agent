#!/usr/bin/env python3
"""
Simple Spotify OAuth test using basic spotipy approach.
"""

import os
import sys
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

def test_simple_oauth():
    """Test using the simplest possible spotipy OAuth approach."""
    
    load_dotenv()
    
    print("🎵 Simple Spotify OAuth Test")
    print("=" * 50)
    
    # Get credentials
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
    
    print(f"Client ID: {client_id[:10]}...{client_id[-5:]}")
    print(f"Redirect URI: {redirect_uri}")
    print()
    
    # Create auth manager with minimal scopes
    scopes = "user-read-playback-state user-modify-playback-state"
    
    print("1. Creating simple OAuth manager...")
    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope=scopes,
        cache_path=".spotify_cache_simple",
        show_dialog=True,  # Force re-authorization
        open_browser=False
    )
    
    print("2. Getting fresh auth URL...")
    auth_url = auth_manager.get_authorize_url()
    print(f"\n🔗 Visit this URL: {auth_url}")
    print("\nAfter clicking 'Agree', paste the callback URL here:")
    
    # Get the callback URL
    callback_url = input("Callback URL: ").strip()
    
    if not callback_url.startswith("http://127.0.0.1:8080/callback?code="):
        print("❌ Invalid callback URL format")
        return False
    
    print("3. Extracting code and getting token...")
    
    try:
        # Try the simplest approach - let spotipy handle everything
        token_info = auth_manager.get_access_token(callback_url, as_dict=True)
        
        if token_info and 'access_token' in token_info:
            print("✅ Token received!")
            
            # Test with Spotify client
            print("4. Testing Spotify API...")
            sp = spotipy.Spotify(auth=token_info['access_token'])
            
            # Get user info (simple test)
            user = sp.current_user()
            print(f"   Logged in as: {user.get('display_name', user.get('id'))}")
            
            # Get devices
            devices_result = sp.devices()
            devices = devices_result.get('devices', []) if devices_result else []
            print(f"   Found {len(devices)} devices:")
            for device in devices:
                status = "🟢 Active" if device.get('is_active') else "⚪ Inactive"
                print(f"     • {device['name']} ({device['type']}) - {status}")
            
            return True
        else:
            print("❌ No token received")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Try alternative method
        print("\n5. Trying alternative approach...")
        try:
            # Extract code manually and use different method
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(callback_url)
            params = parse_qs(parsed.query)
            code = params['code'][0]
            
            print(f"   Code: {code[:20]}...")
            
            # Try get_access_token with just the code
            token_info = auth_manager.get_access_token(code=code, as_dict=True)
            
            if token_info and 'access_token' in token_info:
                print("✅ Alternative method worked!")
                
                sp = spotipy.Spotify(auth=token_info['access_token'])
                user = sp.current_user()
                print(f"   Logged in as: {user.get('display_name', user.get('id'))}")
                return True
            else:
                print("❌ Alternative method also failed")
                return False
                
        except Exception as e2:
            print(f"❌ Alternative method error: {e2}")
            return False

if __name__ == "__main__":
    success = test_simple_oauth()
    print()
    if success:
        print("🎉 OAuth successful! Spotify is ready.")
    else:
        print("❌ OAuth failed.")
    
    sys.exit(0 if success else 1) 