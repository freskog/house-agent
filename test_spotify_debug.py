#!/usr/bin/env python3
"""
Debug script to test Spotify functionality and see why stop music might be failing.
"""

import asyncio
from spotify.tools import create_spotify_tools, _get_spotify_client

async def debug_spotify():
    """Debug Spotify functionality."""
    print("=== Debugging Spotify Functionality ===")
    
    try:
        # Get Spotify client
        client = _get_spotify_client()
        print(f"✅ Got Spotify client: {client}")
        
        # Check devices
        print("\n=== Checking Available Devices ===")
        devices = client.get_devices()
        print(f"Available devices: {len(devices)}")
        for i, device in enumerate(devices, 1):
            status = "🟢 Active" if device.get('is_active') else "⚪ Inactive"
            print(f"  {i}. {device['name']} ({device['type']}) - {status}")
            print(f"      ID: {device['id']}")
        
        # Check spotifyd device
        print(f"\n=== Checking Spotifyd Device ===")
        spotifyd_device_id = client.get_spotifyd_device_id()
        print(f"Configured spotifyd device name: {client.config.spotifyd_device_name}")
        print(f"Found spotifyd device ID: {spotifyd_device_id}")
        
        # Check current playback
        print(f"\n=== Checking Current Playback ===")
        current = client.get_current_playback()
        if current:
            if current.get('item'):
                track = current['item']
                artist = ', '.join([a['name'] for a in track['artists']])
                song = track['name']
                is_playing = current.get('is_playing', False)
                device_name = current.get('device', {}).get('name', 'Unknown')
                print(f"Currently {'playing' if is_playing else 'paused'}: '{song}' by {artist}")
                print(f"Device: {device_name}")
            else:
                print("No track loaded")
        else:
            print("No current playback")
        
        # Test pause functionality directly
        print(f"\n=== Testing Pause Functionality ===")
        print("Attempting to pause music...")
        
        try:
            # Try with no device ID (auto-detect)
            result1 = client.pause()
            print(f"Pause with auto-detect device: {result1}")
            
            # Try with specific spotifyd device if found
            if spotifyd_device_id:
                result2 = client.pause(device_id=spotifyd_device_id)
                print(f"Pause with spotifyd device: {result2}")
            
            # Try with first active device
            active_devices = [d for d in devices if d.get('is_active')]
            if active_devices:
                result3 = client.pause(device_id=active_devices[0]['id'])
                print(f"Pause with first active device: {result3}")
            
        except Exception as e:
            print(f"❌ Error during pause test: {e}")
            import traceback
            traceback.print_exc()
        
        # Test the actual stop_music tool
        print(f"\n=== Testing stop_music Tool ===")
        tools = create_spotify_tools()
        stop_tool = next((t for t in tools if t.name == "stop_music"), None)
        
        if stop_tool:
            try:
                # Call with no arguments (stop_music expects no input)
                result = stop_tool.invoke("")
                print(f"stop_music tool result: '{result}'")
                if result == "":
                    print("✅ Tool returned empty string (success)")
                else:
                    print(f"❌ Tool returned: {result}")
            except Exception as e:
                print(f"❌ Error calling stop_music tool: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ stop_music tool not found")
        
    except Exception as e:
        print(f"❌ Error in debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_spotify()) 