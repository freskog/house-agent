#!/usr/bin/env python3
"""Test office lights command end-to-end with the actual agent."""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

from agent import make_graph
from nodes.schemas import AgentState
from langchain_core.messages import HumanMessage

async def test_office_lights_end_to_end():
    """Test office lights command through the complete agent."""
    
    print("=== TESTING OFFICE LIGHTS END-TO-END ===")
    
    # Create test state for voice command
    state = AgentState(
        messages=[HumanMessage(content="turn on lights in office")],
        next_action=None,
        target_domain=None,
        audio_server=None,
        current_client=None
    )
    
    print("Sending command: 'turn on lights in office'")
    
    try:
        # Create the agent graph and process the request
        async with make_graph() as graph:
            result = await graph.ainvoke(state)
        
        print(f"\n=== RESULTS ===")
        print(f"Messages: {len(result.get('messages', []))}")
        
        # Print the conversation
        for i, msg in enumerate(result.get('messages', [])):
            print(f"Message {i+1}: {type(msg).__name__}: {msg.content}")
        
        print(f"Next action: {result.get('next_action')}")
        print(f"Target domain: {result.get('target_domain')}")
        
        # Check if we got a response
        messages = result.get('messages', [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content') and last_message.content:
                if "couldn't" in last_message.content.lower() or "failed" in last_message.content.lower():
                    print(f"\n❌ FAIL: Got failure response: '{last_message.content}'")
                else:
                    print(f"\n✅ SUCCESS: Got response: '{last_message.content}'")
            else:
                print(f"\n❌ FAIL: Empty response")
        else:
            print(f"\n❌ FAIL: No messages in result")
            
    except Exception as e:
        print(f"\n❌ FAIL: Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_office_lights_end_to_end()) 