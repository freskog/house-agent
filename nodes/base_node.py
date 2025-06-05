"""
Base node implementation.

This provides a cleaner, more maintainable base class that all nodes
can inherit from, implementing the unified interface pattern.
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from .schemas import AgentState, NodeResponse
import logging

class BaseNode(ABC):
    """
    Base class for all nodes with unified interface.
    
    All nodes inherit from this and implement the same handle_request method,
    eliminating the complex dual-mode behavior of the current system.
    """
    
    def __init__(self, tools: List[BaseTool], node_name: str):
        """
        Initialize base node.
        
        Args:
            tools: List of LangChain tools for this domain
            node_name: Name of the node for logging
        """
        self.tools = tools
        self.node_name = node_name
        self.logger = logging.getLogger(f"house_agent.{node_name.lower()}")
    
    @abstractmethod
    async def handle_request(self, state: AgentState) -> AgentState:
        """
        Handle a request using the unified interface.
        
        ALL nodes implement exactly this same method signature.
        No more dual-mode behavior or complex conditional logic.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        pass
    
    def get_last_user_message(self, state: AgentState) -> Optional[HumanMessage]:
        """
        Get the last user message from state.
        
        Args:
            state: Current agent state
            
        Returns:
            Last HumanMessage or None if not found
        """
        messages = state.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message
        return None
    
    def get_user_request_from_messages(self, messages: List) -> str:
        """
        Extract the current user request from messages.
        
        This intelligently determines what the user is asking for,
        whether it's the original request or an agent-generated step.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Current request string
        """
        # Look for the most recent human message
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content
        
        return ""
    
    def can_handle_completely(self, user_request: str) -> bool:
        """
        Determine if this specialist can handle the request completely.
        
        This replaces the complex escalation logic with a simple check.
        With embeddings-based routing, this is mostly unused but kept
        for compatibility.
        
        Args:
            user_request: User's request
            
        Returns:
            True if can handle completely, False if needs escalation
        """
        # Default implementation - with embeddings routing, this is mostly unused
        return True
    
    async def execute_tools_and_respond(self, user_request: str, state: AgentState) -> AIMessage:
        """
        Execute tools and create response for a request.
        
        This is the core specialist functionality - each specialist
        implements this to handle their domain-specific requests.
        
        Args:
            user_request: Request to handle
            state: Current state (for tool execution context)
            
        Returns:
            AI response message
        """
        # Default implementation - subclasses should override for actual tool execution
        return AIMessage(content=f"Handled request: {user_request}")
    
    def create_response(
        self, 
        messages: List[AIMessage], 
        next_action: str = "end",
        target_domain: Optional[str] = None
    ) -> AgentState:
        """
        Create a standardized response in the unified format.
        
        Args:
            messages: Response messages to add
            next_action: What to do next ("end", "escalate", "continue")
            target_domain: Domain to route to if continuing
            
        Returns:
            Updated state with response
        """
        return {
            "messages": messages,
            "next_action": next_action,
            "target_domain": target_domain
        }
    
    def create_escalation_response(
        self, 
        reason: str, 
        domains: List[str], 
        original_request: str
    ) -> AgentState:
        """
        Create a standardized escalation response.
        
        Args:
            reason: Why escalation is needed
            domains: Domains involved
            original_request: Original user request
            
        Returns:
            State that will route to agent
        """
        return {
            "messages": [],  # No response messages for escalation
            "next_action": "escalate",
            "target_domain": "agent"
        }
    
    def detect_multiple_domains(self, user_request: str) -> List[str]:
        """
        Detect if request involves multiple domains.
        
        Args:
            user_request: User's request
            
        Returns:
            List of domains detected in the request
        """
        request_lower = user_request.lower()
        detected_domains = []
        
        # Music keywords
        music_keywords = ["play", "music", "song", "volume", "spotify", "pause", "stop"]
        if any(keyword in request_lower for keyword in music_keywords):
            detected_domains.append("music")
        
        # House keywords  
        house_keywords = ["light", "lights", "temperature", "heat", "cool", "turn on", "turn off"]
        if any(keyword in request_lower for keyword in house_keywords):
            detected_domains.append("house")
        
        # Search keywords
        search_keywords = ["search", "find", "weather", "news", "what is", "who is"]
        if any(keyword in request_lower for keyword in search_keywords):
            detected_domains.append("search")
        
        return detected_domains
    
    def is_agent_instruction(self, message: HumanMessage) -> bool:
        """
        Check if a human message is an agent instruction to a specialist.
        
        Args:
            message: Human message to check
            
        Returns:
            True if this is an agent instruction
        """
        content = message.content
        return content.startswith(("Execute", "Search for:", "Play:", "Set volume", "Turn on", "Turn off"))
    
    def extract_instruction_params(self, message: HumanMessage) -> Dict[str, Any]:
        """
        Extract parameters from an agent instruction message.
        
        Args:
            message: Agent instruction message
            
        Returns:
            Dictionary of extracted parameters
        """
        content = message.content
        params = {}
        
        if content.startswith("Set volume to ") and content.endswith("%"):
            # Extract volume percentage
            volume_str = content[len("Set volume to "):-1]
            try:
                params["volume"] = int(volume_str)
            except ValueError:
                pass
        elif content.startswith("Search for: "):
            params["query"] = content[len("Search for: "):]
        elif content.startswith("Play: "):
            params["query"] = content[len("Play: "):]
        
        return params 