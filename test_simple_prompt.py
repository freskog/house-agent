#!/usr/bin/env python3
"""Test with a minimal prompt to see if the LLM can generate tool calls."""

import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from nodes.schemas import SpecialistResponse

async def test_minimal_prompt():
    """Test the LLM with a very simple prompt."""
    
    # Create LLM with structured output using function calling
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0
    ).with_structured_output(SpecialistResponse, method="function_calling")
    
    prompt = """User wants to "turn on lights in office".

Available entity: "Fredrik's office ceiling light"

Generate SpecialistResponse with:
- action: "execute_tools"
- tool_calls: [{"name": "HassTurnOn", "arguments": {"name": "Fredrik's office ceiling light"}}]
- needs_tool_results: false
- success_response: "Office lights turned on"
- failure_response: "Failed to turn on office lights"

Fill in ALL fields correctly."""

    print("=== MINIMAL PROMPT TEST ===")
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    
    print(f"Response: {response}")
    print(f"Tool calls: {response.tool_calls}")
    
    if response.tool_calls and len(response.tool_calls) > 0:
        tool_call = response.tool_calls[0]
        if tool_call and "name" in tool_call and "arguments" in tool_call:
            print("✅ SUCCESS: Proper tool call generated!")
            print(f"Tool name: {tool_call['name']}")
            print(f"Tool args: {tool_call['arguments']}")
        else:
            print("❌ FAIL: Tool call has wrong format")
    else:
        print("❌ FAIL: No tool calls generated")

if __name__ == "__main__":
    asyncio.run(test_minimal_prompt()) 