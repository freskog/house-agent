"""
House Agent - Entry Point

This module provides the main entry point for the house agent system.
It builds a clean, maintainable architecture with:

- Simplified Router: Fast keyword-based routing with unified interface
- Simplified Agent: Message-based planning and orchestration  
- Hybrid Specialists: Current specialist nodes with simplified interface adapters

External Interface:
- make_graph(): Async context manager returning compiled LangGraph (same interface as before)
- SimpleAgentState: Simplified state type (5 fields vs 18+ before)

Key Improvements:
- 72% reduction in state complexity
- Unified routing function (1 vs 4 functions)
- Message-based execution (eliminates serialization bugs)
- Consistent node interfaces
- 10-20% performance improvement expected
"""

from contextlib import asynccontextmanager
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any
import os
import time
import logging

# Load environment variables first
from dotenv import load_dotenv
load_dotenv(override=True)

# Import simplified architecture components
from nodes.schemas import AgentState, route_next_node
from nodes.router_node import RouterNode
from nodes.agent_node import AgentNode

# Import current specialist nodes
from nodes import MusicNode, HouseNode, SearchNode, ClarificationNode

# Lightweight hybrid adapter for specialist nodes
class HybridNode:
    """Wrapper that allows current nodes to work with simplified state"""
    
    def __init__(self, current_node, node_name: str):
        self.current_node = current_node
        self.node_name = node_name
    
    async def handle_request(self, state: AgentState) -> AgentState:
        """Handle request using current node but with simplified interface"""
        try:
            # Create minimal current state format for compatibility
            current_state = {
                "messages": state.get("messages", []),
                "audio_server": state.get("audio_server"),
                "current_client": state.get("current_client"),
                "needs_clarification": False,
                "original_request": None,
                "clarification_count": 0,
                "current_domain": state.get("target_domain", "router"),
            }
            
            # Map simplified fields to current fields for routing
            next_action = state.get("next_action")
            target_domain = state.get("target_domain")
            
            if next_action == "escalate":
                current_state["escalation_context"] = {
                    "from_domain": target_domain or "unknown",
                    "escalation_reason": "Multi-domain request detected",
                    "detected_domains": [target_domain] if target_domain else [],
                    "user_request": self._get_user_request(state)
                }
            elif next_action == "continue":
                current_state["route_destination"] = target_domain
                current_state["agent_next_step"] = target_domain
            
            # Call the current node
            result = await self.current_node.handle_request(current_state)
            
            # Convert back to simplified format
            return {
                "messages": result.get("messages", state.get("messages", [])),
                "next_action": "end",  # Most specialists end after execution
                "target_domain": None,
                "audio_server": state.get("audio_server"),
                "current_client": state.get("current_client")
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid {self.node_name} node: {e}")
            # Fallback to escalation
            return {
                "messages": state.get("messages", []),
                "next_action": "escalate",
                "target_domain": "agent",
                "audio_server": state.get("audio_server"),
                "current_client": state.get("current_client")
            }
    
    def _get_user_request(self, state: AgentState) -> str:
        """Extract user request from messages"""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if hasattr(message, 'content'):
                return message.content
        return ""

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("house_agent")

# Set up in-memory caching for LangChain
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
set_llm_cache(InMemoryCache())
logger.info("Configured LangChain cache")

# Initialize LangSmith client only if enabled  
if os.environ.get("LANGSMITH_TRACING_V2", "false").lower() == "true":
    from langsmith import Client
    logger.info("LangSmith client initialized")
else:
    logger.info("LangSmith tracing disabled")

# Global variables to prevent multiple initializations
_graph_initialized = False
_cached_graph = None

@asynccontextmanager
async def make_graph():
    """
    Create and return the house agent graph.
    
    This is the main entry point used by external systems.
    Returns a compiled LangGraph with streamlined architecture.
    
    Architecture:
    - Router → {Music, House, Search, Agent} (unified routing)
    - Agent → {Music, House, Search, Clarification} (message-based)
    - All nodes use consistent AgentState interface
    
    Key Features:
    - Streamlined state (5 fields)
    - Single unified routing function
    - Consistent node interfaces
    - Message-based execution
    """
    global _graph_initialized, _cached_graph
    
    # Return cached graph if already initialized
    if _graph_initialized and _cached_graph is not None:
        logger.info("Returning cached graph")
        yield _cached_graph
        return
    
    logger.info("Initializing agent system...")
    start_time = time.time()
    
    try:
        # Initialize current specialist nodes
        logger.info("Initializing specialist nodes...")
        current_music = MusicNode()
        current_house = HouseNode()
        current_search = SearchNode()
        current_clarification = ClarificationNode()
        
        # Wrap current specialists in hybrid adapters for interface compatibility
        music_node = HybridNode(current_music, "Music")
        house_node = HybridNode(current_house, "House")
        search_node = HybridNode(current_search, "Search")
        clarification_node = HybridNode(current_clarification, "Clarification")
        
        # Initialize router and agent
        router_node = RouterNode()
        
        specialists = {
            "music": music_node,
            "house": house_node,
            "search": search_node,
            "clarification": clarification_node
        }
        agent_node = AgentNode(specialists)
        
        logger.info("Building graph...")
        
        # Build the state graph
        builder = StateGraph(AgentState)
        
        # Add all nodes
        builder.add_node("router", router_node.handle_request)
        builder.add_node("music", music_node.handle_request)
        builder.add_node("house", house_node.handle_request)
        builder.add_node("search", search_node.handle_request)
        builder.add_node("clarification", clarification_node.handle_request)
        builder.add_node("agent", agent_node.handle_request)
        
        # Set entry point
        builder.add_edge(START, "router")
        
        # Use unified routing function throughout - this replaces ALL the complex routing logic
        routing_map = {
            "music": "music",
            "house": "house",
            "search": "search",
            "agent": "agent",
            "clarification": "clarification",
            "END": END
        }
        
        # Apply the same unified routing to ALL nodes
        for node_name in ["router", "music", "house", "search", "clarification", "agent"]:
            builder.add_conditional_edges(node_name, route_next_node, routing_map)
        
        # Compile the graph
        graph = builder.compile(debug=True)
        
                # Cache and log completion
        _graph_initialized = True
        _cached_graph = graph
        
        init_time = time.time() - start_time
        logger.info(f"Agent system initialized in {init_time:.2f}s")
        logger.info("Architecture: Router → {Music, House, Search, Agent} | Agent → {Music, House, Search, Clarification}")
        logger.info("Key features: Streamlined state (5 fields), unified routing, message-based execution")
        
        yield graph
        
    except Exception as e:
        logger.error(f"Error initializing agent: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage
    
    async def test_agent():
        """Test the agent system."""
        async with make_graph() as graph:
            logger.info("Agent graph created successfully")
            
            # Test simple music request (router → music)
            test_state = {
                "messages": [HumanMessage(content="play some jazz music")]
            }
            
            logger.info("Testing simple music request...")
            result = await graph.ainvoke(test_state)
            logger.info("Test completed successfully")
            
            # Test composite request (router → agent → music + house)
            composite_state = {
                "messages": [HumanMessage(content="play jazz and turn on lights")]
            }
            
            logger.info("Testing composite request...")
            result = await graph.ainvoke(composite_state)
            logger.info("Composite test completed successfully")
            
    asyncio.run(test_agent()) 