"""
Node-based architecture for the house agent.

This package contains domain-specific nodes that handle different types of requests:
- MusicNode: Music and audio control  
- HouseNode: Home automation and device control
- SearchNode: Web search and information retrieval
- ClarificationNode: User clarification and disambiguation

STREAMLINED ARCHITECTURE:
- RouterNode: Unified keyword-based routing
- AgentNode: Message-based multi-domain orchestration
- BaseNode: Consistent interface for all nodes
- AgentState: Streamlined state (5 fields vs 18+)

All nodes inherit shared infrastructure and use consistent interfaces.
"""

# Current specialist nodes (still used via hybrid adapters)
from .music_node import MusicNode
from .house_node import HouseNode
from .search_node import SearchNode
from .clarification_node import ClarificationNode

# Streamlined architecture components (ACTIVE)
from .schemas import AgentState, NodeResponse, route_next_node
from .base_node import BaseNode
from .router_node import RouterNode
from .agent_node import AgentNode

__all__ = [
    # ACTIVE: Specialist nodes
    "MusicNode",
    "HouseNode", 
    "SearchNode",
    "ClarificationNode",
    
    # ACTIVE: Streamlined architecture
    "AgentState",
    "BaseNode",
    "RouterNode", 
    "AgentNode",
    "NodeResponse",
    "route_next_node",
] 