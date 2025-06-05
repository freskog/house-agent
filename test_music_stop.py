#!/usr/bin/env python3
"""
Test script to verify music stop functionality in hierarchical agent.
"""

import asyncio
from agent import make_graph
from langchain_core.messages import HumanMessage

async def test_music_stop():
    """Test the music stop functionality."""
    print("=== Testing Music Stop Functionality ===")
    
    async with make_graph() as graph:
        # Test stop music command
        test_state = {
            'messages': [HumanMessage(content='stop music')]
        }
        
        print("Query: 'stop music'")
        print("Expected: Router → Music → Stop tool execution → Silent success")
        
        result = await graph.ainvoke(test_state)
        
        print(f"\n=== Results ===")
        print(f"Final messages count: {len(result['messages'])}")
        
        # Check final message
        if result['messages']:
            final_message = result['messages'][-1]
            print(f"Final message content: '{final_message.content}'")
            
            if final_message.content == "":
                print("✅ SUCCESS: Music stopped silently (empty response)")
            elif "Failed to stop" in final_message.content:
                print("❌ FAILURE: Music stop command failed")
            else:
                print(f"⚠️  UNEXPECTED: Got response: '{final_message.content}'")
        
        # Show routing info if available
        if 'current_domain' in result:
            print(f"Final domain: {result['current_domain']}")

if __name__ == "__main__":
    asyncio.run(test_music_stop()) 