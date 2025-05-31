#!/usr/bin/env python3
"""
Debug script to examine Spotify authorization URL generation.
"""

from spotify import SpotifyClient
from dotenv import load_dotenv
import os
from urllib.parse import urlparse, parse_qs

def debug_spotify_url():
    load_dotenv()
    
    print('=== DEBUG: Spotify Configuration ===')
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
    
    print(f'Client ID: {client_id}')
    print(f'Client Secret: {"SET" if client_secret else "NOT SET"}')
    print(f'Redirect URI: {redirect_uri}')
    print()
    
    print('=== DEBUG: Creating Spotify Client ===')
    try:
        client = SpotifyClient()
        print('✅ Client created successfully')
    except Exception as e:
        print(f'❌ Error creating client: {e}')
        return
    
    print()
    print('=== DEBUG: Authorization URL ===')
    try:
        auth_url = client.get_auth_url()
        print(f'Generated URL: {auth_url}')
        print()
        
        print('=== DEBUG: URL Components ===')
        parsed = urlparse(auth_url)
        print(f'Base URL: {parsed.scheme}://{parsed.netloc}{parsed.path}')
        
        params = parse_qs(parsed.query)
        for key, value in params.items():
            print(f'{key}: {value[0] if value else "None"}')
            
        # Check for common issues
        print()
        print('=== DEBUG: Validation ===')
        
        if 'client_id' not in params:
            print('❌ Missing client_id parameter')
        elif params['client_id'][0] != client_id:
            print('❌ client_id mismatch')
        else:
            print('✅ client_id looks good')
            
        if 'redirect_uri' not in params:
            print('❌ Missing redirect_uri parameter')
        elif params['redirect_uri'][0] != redirect_uri:
            print(f'❌ redirect_uri mismatch: {params["redirect_uri"][0]} vs {redirect_uri}')
        else:
            print('✅ redirect_uri looks good')
            
        if 'response_type' not in params:
            print('❌ Missing response_type parameter')
        elif params['response_type'][0] != 'code':
            print(f'❌ Wrong response_type: {params["response_type"][0]}')
        else:
            print('✅ response_type looks good')
            
        if 'scope' not in params:
            print('❌ Missing scope parameter')
        else:
            scopes = params['scope'][0].split(' ')
            print(f'✅ scopes: {", ".join(scopes)}')
            
    except Exception as e:
        print(f'❌ Error generating auth URL: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_spotify_url() 