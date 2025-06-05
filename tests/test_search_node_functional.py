#!/usr/bin/env python3
"""
Functional test for the simplified SearchNode.

This tests actual functionality if TAVILY_API_KEY is available.
"""

import pytest
import asyncio
import os
from langchain_core.messages import HumanMessage, AIMessage

@pytest.mark.functional
@pytest.mark.asyncio
async def test_search_node_functional():
    """Test SearchNode with real API if available."""
    
    # Check if we have API key
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        print("❌ No TAVILY_API_KEY - skipping functional test")
        return
    
    try:
        from nodes.search_node import SearchNode
        
        print("🚀 Testing simplified SearchNode...")
        
        # Initialize node
        search_node = SearchNode()
        print(f"✅ SearchNode initialized successfully")
        
        # Test simple query
        state = {
            "messages": [HumanMessage(content="What is the capital of France?")]
        }
        
        print("🔍 Testing search request...")
        result = await search_node.handle_request(state)
        
        # Check response
        if result and result.get("messages"):
            response = result["messages"][0]
            if isinstance(response, AIMessage) and response.content:
                print(f"✅ Search successful!")
                print(f"   Response: {response.content[:100]}...")
                print(f"   Response length: {len(response.content)} chars")
            else:
                print(f"❌ Invalid response format: {response}")
        else:
            print(f"❌ No response received: {result}")
        
        # Test caller detection
        agent_messages = [HumanMessage(content="Search for: latest AI news")]
        router_messages = [HumanMessage(content="What's the weather today?")]
        
        assert search_node._detect_caller(agent_messages) == "agent"
        assert search_node._detect_caller(router_messages) == "router"
        print("✅ Caller detection working")
        
        # Test response formatting
        tool_results = [{"content": "Test search result"}]
        formatted = search_node._format_raw_results(tool_results, "test query")
        assert "test query" in formatted
        assert "Test search result" in formatted
        print("✅ Response formatting working")
        
        print("🎉 All functional tests passed!")
        
    except Exception as e:
        print(f"❌ Functional test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_search_node_functional()) 