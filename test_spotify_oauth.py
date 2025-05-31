#!/usr/bin/env python3
"""
Test script for Spotify OAuth flow.
This will help you authenticate with Spotify for the first time.
"""

import os
import sys
from dotenv import load_dotenv
from spotify import SpotifyClient

def test_oauth_flow():
    """Test the Spotify OAuth authentication flow."""
    
    # Load environment variables
    load_dotenv()
    
    print("🎵 Spotify OAuth Test")
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
        print("Please set SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, and SPOTIFY_REDIRECT_URI in your .env file")
        return False
    
    try:
        # Create Spotify client
        print("1. Creating Spotify client...")
        spotify_client = SpotifyClient()
        
        # Check if already authenticated
        print("2. Checking existing authentication...")
        if spotify_client.is_authenticated():
            print("✅ Already authenticated!")
            
            # Test basic API call
            print("3. Testing API access...")
            devices = spotify_client.get_devices()
            print(f"   Found {len(devices)} Spotify devices:")
            for device in devices:
                status = "🟢 Active" if device.get('is_active') else "⚪ Inactive"
                print(f"   • {device['name']} ({device['type']}) - {status}")
            
            return True
        
        else:
            print("🔐 Authentication required")
            print("3. Getting authorization URL...")
            auth_url = spotify_client.get_auth_url()
            
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
                
            # Authenticate with response URL
            print("4. Authenticating with response URL...")
            success = spotify_client.authenticate(response_url)
            
            if success:
                print("✅ Authentication successful!")
                
                # Test basic API call
                print("5. Testing API access...")
                devices = spotify_client.get_devices()
                print(f"   Found {len(devices)} Spotify devices:")
                for device in devices:
                    status = "🟢 Active" if device.get('is_active') else "⚪ Inactive"
                    print(f"   • {device['name']} ({device['type']}) - {status}")
                
                return True
            else:
                print("❌ Authentication failed")
                return False
                
    except Exception as e:
        print(f"❌ Error during OAuth flow: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_oauth_flow()
    print()
    if success:
        print("🎉 OAuth flow completed successfully!")
        print("The agent can now control Spotify playback.")
    else:
        print("❌ OAuth flow failed.")
        print("Please check your credentials and try again.")
    
    sys.exit(0 if success else 1) 