#!/usr/bin/env python3
"""
Functional testing for SearchNode to verify real web search functionality.
Tests require TAVILY_API_KEY environment variable.
"""

import os
import pytest
import asyncio
from nodes.search_node import SearchNode, invoke_tools
from langchain_core.messages import HumanMessage
from utils.logging_config import setup_logging

# Setup logging for tests  
logger = setup_logging(__name__)

@pytest.mark.functional
@pytest.mark.asyncio 
async def test_search_node_functional():
    """Test SearchNode with real web search functionality."""
    
    # Skip if no API key configured
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("❌ No TAVILY_API_KEY - skipping functional test")
        pytest.skip("TAVILY_API_KEY not configured")
        return
        
    try:
        logger.info("🚀 Testing simplified SearchNode...")
        
        # Test SearchNode creation 
        search_node = SearchNode()
        logger.info("✅ SearchNode initialized successfully")
        
        # Test basic search
        state = {
            "messages": [HumanMessage(content="What is the weather in Stockholm today?")],
        }
        logger.info("🔍 Testing search request...")
        
        result = await search_node.ainvoke(state)
        
        if result and result.get("messages"):
            response = result["messages"][-1]
            logger.info("✅ Search successful!")
            logger.info(f"   Response: {response.content[:100]}...")
            logger.info(f"   Response length: {len(response.content)} chars")
        else:
            logger.error("❌ Invalid response format: {response}")
            assert False, "Invalid response format"
        
        if not result.get("messages"):
            logger.error("❌ No response received: {result}")
            assert False, "No response received"
        
        # Test the tools directly
        tools = invoke_tools([])
        
        # Test that caller detection works
        assert any("tavily" in str(tool).lower() for tool in tools)
        logger.info("✅ Caller detection working")
        
        # Test that search results are properly formatted
        final_response = result["messages"][-1]
        assert len(final_response.content) > 50  # Should have substantial content
        logger.info("✅ Response formatting working")
        
        logger.info("🎉 All functional tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Functional test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_search_node_functional()) 