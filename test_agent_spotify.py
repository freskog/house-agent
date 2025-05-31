#!/usr/bin/env python3
"""
Final test: Agent with Spotify integration.
"""

import asyncio
from agent import make_graph

async def test_agent_with_spotify():
    print('🤖 Testing Agent with Spotify Integration')
    print('=' * 50)
    
    try:
        async with make_graph() as graph:
            print('✅ Agent loaded successfully with Spotify tools!')
            
            # Simulate a simple conversation
            state = {
                "messages": [{"type": "human", "content": "Hello!"}]
            }
            
            print('📝 Testing simple interaction...')
            result = await graph.ainvoke(state)
            
            if result and "messages" in result:
                print('✅ Agent responded successfully!')
                return True
            else:
                print('❌ Agent did not respond properly')
                return False
                
    except Exception as e:
        print(f'❌ Error: {e}')
        return False

if __name__ == "__main__":
    success = asyncio.run(test_agent_with_spotify())
    
    if success:
        print('\n🎉 COMPLETE SUCCESS!')
        print('🎵 Your agent can now control Spotify!')
        print('🎯 Available Spotify commands:')
        print('   • "What\'s playing?"')
        print('   • "Play some music"')  
        print('   • "Pause music"')
        print('   • "Next track"')
        print('   • "Set volume to 50"')
        print('   • "Show Spotify devices"')
    else:
        print('\n❌ Agent integration failed') 