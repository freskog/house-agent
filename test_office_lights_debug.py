#!/usr/bin/env python3
"""Debug the office lights prompt issue."""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from nodes.house_node import HouseNode
from nodes.schemas import AgentState
from langchain_core.messages import SystemMessage

async def test_office_lights_prompt():
    """Test the current prompt for office lights command."""
    
    # Create house node
    house_node = HouseNode()
    
    # Initialize
    await house_node.initialize_mcp_client()
    await house_node.load_tools()
    await house_node.load_homeassistant_prompt()
    
    print(f"Tools loaded: {len(house_node.tools) if house_node.tools else 0}")
    print(f"HA Content length: {len(house_node.homeassistant_content) if house_node.homeassistant_content else 0}")
    
    if house_node.homeassistant_content:
        print("\n=== FULL HOME ASSISTANT CONTEXT ===")
        print(house_node.homeassistant_content)
        
        # Look for office-related entities
        print("\n=== OFFICE-RELATED ENTITIES ===")
        lines = house_node.homeassistant_content.split('\n')
        for line in lines:
            if 'office' in line.lower() and 'light' in line.lower():
                print(f"FOUND OFFICE LIGHT: {line}")
    
    # Show available tools
    print(f"\n=== AVAILABLE TOOLS ===")
    if house_node.tools:
        for tool in house_node.tools[:5]:  # Show first 5
            print(f"Tool: {tool.name}")
            if hasattr(tool, 'description'):
                print(f"  Description: {tool.description}")
    
    # Create test state
    state = AgentState(
        messages=[],
        current_step=1,
        max_steps=3,
        user_intent="turn on lights in office",
        context={}
    )
    
    # Test the raw LLM call
    print("\n=== TESTING RAW LLM CALL ===")
    try:
        current_date = "December 23, 2024"
        ha_context = house_node.get_homeassistant_context()
        user_request = "turn on lights in office"
        called_by = "agent"
        
        prompt = f"""You are a home automation specialist. Today is {current_date}.

User request: "{user_request}"
Called by: {called_by}

{ha_context}

IMPORTANT ENTITY MATCHING INSTRUCTIONS:
When the user asks to control devices, you MUST match their request to the exact entity names from the Home Assistant context above.

Entity Matching Rules:
1. "lights in office" / "office lights" → Look for entities with "office" in the name
2. "living room lights" → Look for entities with "living room" in the name  
3. "bedroom temperature" → Look for thermostats/sensors with "bedroom" in the name
4. "front door lock" → Look for locks with "front door" in the name

Available Tools:
- HassTurnOn(name="exact_entity_name") - Turn on a device
- HassTurnOff(name="exact_entity_name") - Turn off a device
- HassSetTemperature(name="exact_entity_name", temperature=X) - Set temperature
- (Check context for full list of available tools)

Domain Detection Rules:
- If request involves ONLY home automation (lights, temperature, etc.) → this is purely house domain
- If request involves home automation AND other domains (music, search, etc.) → escalate if called_by="router"
- If called_by="agent" → never escalate, just handle house automation part

Choose ONE action:

1. execute_tools: For home automation requests needing device control
   - CRITICAL: Extract exact entity names from Home Assistant context above
   - Set needs_tool_results=false for simple commands (turn on lights)
   - Set needs_tool_results=true if you need to see results (device status queries)
   - Provide success_response and failure_response for fast path
   - tool_calls format: [{{"name": "HassTurnOn", "arguments": {{"name": "exact_entity_name"}}}}]

2. respond: For general information about home automation or explanations

3. escalate: ONLY if called_by="router" AND request involves multiple domains
   - Detect all domains involved
   - Explain why escalation is needed

Examples:
- "turn on the living room lights" → execute_tools with HassTurnOn(name="Living Room Light")
- "what's the temperature in bedroom" → execute_tools with needs_tool_results=true
- "turn on lights and play music" → escalate if called_by="router" (multi-domain)
- "how do smart lights work" → respond (no tools needed)

CRITICAL: When generating tool_calls, use the EXACT entity names from the Home Assistant context above!

For this request "turn on lights in office", I can see "Fredrik's office ceiling light" in the context.
Generate tool_calls: [{{"name": "HassTurnOn", "arguments": {{"name": "Fredrik's office ceiling light"}}}}]

Respond with structured output following SpecialistResponse schema."""

        print("=== SENDING PROMPT TO LLM ===")
        response = await house_node.llm.ainvoke([SystemMessage(content=prompt)])
        print(f"LLM Response type: {type(response)}")
        print(f"LLM Response: {response}")
        
        if hasattr(response, 'content'):
            print(f"Response content: {response.content}")
        if hasattr(response, 'tool_calls'):
            print(f"Response tool_calls: {response.tool_calls}")
        
        # Try to parse as SpecialistResponse
        try:
            from nodes.schemas import SpecialistResponse
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print("Response has tool_calls - this is function calling format")
            else:
                print("Response is text format - needs parsing")
                
        except Exception as parse_e:
            print(f"Parse error: {parse_e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_office_lights_prompt()) 