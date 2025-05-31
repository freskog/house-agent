#!/usr/bin/env python3
"""
Test our Spotify client with the new authentication.
"""

from spotify import SpotifyClient
from dotenv import load_dotenv

def test_spotify_client():
    load_dotenv()
    print('🎵 Testing Spotify Client Integration')
    print('=' * 50)

    try:
        print('1. Creating Spotify client...')
        client = SpotifyClient()
        
        print(f'2. Checking authentication: {client.is_authenticated()}')
        
        if client.is_authenticated():
            print('3. Getting devices...')
            devices = client.get_devices()
            print(f'   Found {len(devices)} devices:')
            for device in devices:
                status = '🟢 Active' if device.get('is_active') else '⚪ Inactive'
                print(f'     • {device["name"]} ({device["type"]}) - {status}')
                
            print('4. Checking current playback...')
            current = client.get_current_playback()
            if current and current.get('item'):
                track = current['item']
                artist = ', '.join([a['name'] for a in track['artists']])
                print(f'   Currently playing: {track["name"]} by {artist}')
            else:
                print('   Nothing currently playing')
                
            print('✅ Spotify client is working!')
            return True
        else:
            print('❌ Not authenticated')
            return False
            
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_spotify_client()
    if success:
        print('\n🎉 Spotify integration is ready!')
        print('The agent can now control Spotify playback.')
    else:
        print('\n❌ Spotify integration failed.') 