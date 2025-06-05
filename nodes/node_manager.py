"""
Node Manager for orchestrating domain-specific nodes.

This module provides a centralized manager that:
- Initializes all domain-specific nodes
- Routes requests to appropriate nodes based on content analysis
- Manages inter-node communication and state
- Maintains backward compatibility with the original agent interface
"""

from typing import Dict, Any, Optional, List, Union
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import BaseTool
from .base_node import BaseNode
from .schemas import AgentState
from .music_node import MusicNode
from .house_node import HouseNode
from .search_node import SearchNode
from .clarification_node import ClarificationNode
import logging
import asyncio

logger = logging.getLogger(__name__)

class NodeManager:
    """
    Central orchestrator for all domain-specific nodes.
    
    This class manages the lifecycle and interaction between different nodes
    while maintaining the same external interface as the original agent.
    """
    
    def __init__(self):
        """Initialize the node manager and all domain nodes synchronously."""
        logger.info("Initializing Node Manager...")
        
        # Initialize all domain nodes (now all synchronous)
        self.nodes: Dict[str, BaseNode] = {
            "music": MusicNode(),
            "house": HouseNode(),
            "search": SearchNode(),
            "clarification": ClarificationNode()
        }
        
        # Collect tools immediately available (Spotify, WebSearch, Clarification)
        self.all_tools = self._collect_all_tools()
        
        logger.info(f"Initialized {len(self.nodes)} domain nodes with {len(self.all_tools)} immediate tools")
        logger.info("House automation tools will be loaded lazily on first use")
    
    def _collect_all_tools(self) -> List[BaseTool]:
        """Collect all tools from all nodes into a single list."""
        all_tools = []
        
        for node_name, node in self.nodes.items():
            if node.tools:
                all_tools.extend(node.tools)
                logger.debug(f"Added {len(node.tools)} tools from {node_name} node")
        
        logger.debug(f"Collected {len(all_tools)} total tools from all nodes")
        return all_tools
    
    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all tools from all nodes, including lazy-loaded ones.
        
        Returns:
            List of all available tools
        """
        # Always recollect tools to include any lazy-loaded ones
        return self._collect_all_tools()
    
    def detect_domain(self, message: str) -> str:
        """
        Detect which domain should handle the request.
        
        Args:
            message: User message to analyze
            
        Returns:
            Domain name ("music", "house", "search", "clarification")
        """
        # Priority order for domain detection
        # (More specific domains first to avoid conflicts)
        
        # Check Music first (music is very specific)
        if self.nodes["music"].should_handle_request(message):
            return "music"
        
        # Check House (device control is specific)
        if self.nodes["house"].should_handle_request(message):
            return "house"
        
        # Check Search (questions and lookups)
        if self.nodes["search"].should_handle_request(message):
            return "search"
        
        # Check Clarification (unclear/ambiguous requests)
        if self.nodes["clarification"].should_handle_request(message):
            return "clarification"
        
        # Default fallback - try search for unknown requests
        return "search"
    
    def get_node(self, domain: str) -> Optional[BaseNode]:
        """
        Get a specific domain node.
        
        Args:
            domain: Domain name
            
        Returns:
            Domain node or None if not found
        """
        return self.nodes.get(domain)
    

    
    def should_end_silently(self, state: AgentState) -> bool:
        """
        Check if the last tool execution should end silently.
        
        This is primarily used for Spotify commands that shouldn't trigger TTS.
        
        Args:
            state: Current agent state
            
        Returns:
            True if should end silently, False otherwise
        """
        # Check with Music node first (most likely to have silent commands)
        music_node = self.nodes.get("music")
        if music_node and music_node.should_end_silently(state):
            return True
        
        # Other nodes could implement silent behavior if needed
        return False
    
    def get_homeassistant_context(self) -> str:
        """
        Get house automation context for system prompts.
        
        Returns:
            House automation context string or empty string if not available
        """
        house_node = self.nodes.get("house")
        if house_node:
            return house_node.get_homeassistant_context()
        return ""
    
    def get_current_song(self) -> str:
        """
        Get current song from Music node.
        
        Returns:
            Current song string or default message
        """
        music_node = self.nodes.get("music")
        if music_node:
            return music_node.current_song
        return "Nothing playing"
    
    def is_in_clarification_mode(self, state: AgentState) -> bool:
        """
        Check if we're currently in clarification mode.
        
        Args:
            state: Current agent state
            
        Returns:
            True if in clarification mode
        """
        clarification_node = self.nodes.get("clarification")
        if clarification_node:
            return clarification_node.is_in_clarification_mode(state)
        return state.get("needs_clarification", False)
    
    def handle_clarification_response(self, state: AgentState) -> Dict[str, Any]:
        """
        Handle user's response to a clarification question.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state based on response
        """
        clarification_node = self.nodes.get("clarification")
        if clarification_node:
            return clarification_node.handle_clarification_response(state)
        
        # Fallback behavior
        return {"needs_clarification": False, "clarification_count": 0}
    
    async def route_to_domain(self, state: AgentState, domain: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Route a request to a specific domain node.
        
        Args:
            state: Current agent state
            domain: Target domain name
            config: Optional configuration
            
        Returns:
            Updated state from the domain node
        """
        node = self.get_node(domain)
        if not node:
            logger.error(f"Domain node '{domain}' not found")
            return state
        
        try:
            return await node.handle_request(state, config)
        except Exception as e:
            logger.error(f"Error handling request in {domain} node: {e}")
            # Return error response
            return node.create_response([
                AIMessage(content=f"Sorry, I encountered an error while processing your {domain} request.")
            ])
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about available nodes and tools.
        
        Returns:
            Dictionary with routing statistics
        """
        all_tools = self.get_all_tools()
        stats = {
            "total_nodes": len(self.nodes),
            "total_tools": len(all_tools),
            "nodes": {}
        }
        
        for domain, node in self.nodes.items():
            stats["nodes"][domain] = {
                "tools_count": len(node.tools),
                "node_name": node.node_name,
                "available": len(node.tools) > 0 or domain == "clarification"
            }
        
        return stats 