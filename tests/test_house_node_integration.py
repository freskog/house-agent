#!/usr/bin/env python3
"""
Integration test for house automation via office lights question.
"""

import pytest
import asyncio
from langchain_core.messages import HumanMessage
from agent import make_graph
from utils.logging_config import setup_logging

# Setup logging for tests
logger = setup_logging(__name__)

@pytest.mark.integration
@pytest.mark.asyncio
async def test_office_lights():
    """Test the office lights question."""
    logger.info("🏠 Testing office lights question...")
    
    try:
        async with make_graph() as graph:
            initial_state = {
                "messages": [HumanMessage(content="Are the lights on in fredriks office?")],
                "next_action": "continue",
                "target_domain": None,
                "audio_server": None,
                "current_client": None
            }
            
            final_state = await graph.ainvoke(initial_state, {"recursion_limit": 10})
            
            if final_state and final_state.get("messages"):
                last_message = final_state["messages"][-1]
                logger.info(f"✅ Response: {last_message.content}")
            else:
                logger.warning("❌ No response received")
                
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_office_lights()) 