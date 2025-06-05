"""
Router node implementation.

This router uses unified routing logic with keyword-based intent detection.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from .base_node import BaseNode
from .schemas import AgentState
import logging

logger = logging.getLogger(__name__)

class RouterNode(BaseNode):
    """
    Router with unified interface and keyword-based intent detection.
    
    Uses confidence thresholds and domain detection for fast, reliable routing.
    """
    
    def __init__(self):
        """Initialize simplified router (no tools needed)."""
        super().__init__([], "Router")
        self.confidence_threshold = 0.7  # Threshold for routing to specialists vs agent
    
    async def handle_request(self, state: AgentState) -> AgentState:
        """
        Route requests using simplified logic.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with routing decision
        """
        user_message = self.get_last_user_message(state)
        if not user_message:
            return self.create_response(
                [AIMessage(content="I didn't receive a clear request.")],
                next_action="end"
            )
        
        # Classify intent using simplified logic
        intent = await self.classify_intent(user_message.content)
        
        # Route based on intent confidence and domain count
        if intent["confidence"] > self.confidence_threshold and len(intent["domains"]) == 1:
            # High confidence, single domain -> route to specialist
            target_domain = intent["domains"][0]
            return {
                **state,
                "next_action": "continue",
                "target_domain": target_domain
            }
        else:
            # Low confidence or multiple domains -> escalate to agent
            return {
                **state,
                "next_action": "escalate", 
                "target_domain": "agent"
            }
    
    async def classify_intent(self, user_request: str) -> Dict[str, Any]:
        """
        Classify user intent with confidence scoring.
        
        This replaces the complex LLM-based classification with
        simpler keyword-based detection that's faster and more reliable.
        
        Args:
            user_request: User's request text
            
        Returns:
            Dictionary with domains, confidence, and classification
        """
        request_lower = user_request.lower()
        detected_domains = []
        confidence_scores = {}
        
        # Music domain detection
        music_keywords = ["play", "music", "song", "volume", "spotify", "pause", "stop", "next", "previous"]
        music_score = sum(1 for keyword in music_keywords if keyword in request_lower)
        if music_score > 0:
            detected_domains.append("music")
            confidence_scores["music"] = min(music_score / 2.0, 1.0)  # Normalize to 0-1 (higher confidence)
        
        # House domain detection
        house_keywords = ["light", "lights", "temperature", "heat", "cool", "turn on", "turn off", "dim", "bright"]
        house_score = sum(1 for keyword in house_keywords if keyword in request_lower)
        if house_score > 0:
            detected_domains.append("house")
            confidence_scores["house"] = min(house_score / 3.0, 1.0)
        
        # Search domain detection
        search_keywords = ["search", "find", "weather", "news", "what is", "who is", "when is", "where is"]
        search_score = sum(1 for keyword in search_keywords if keyword in request_lower)
        if search_score > 0:
            detected_domains.append("search")
            confidence_scores["search"] = min(search_score / 2.0, 1.0)  # Higher confidence
        
        # Calculate overall confidence
        if not detected_domains:
            # No clear domain detected
            overall_confidence = 0.0
            detected_domains = ["clarification"]
        elif len(detected_domains) == 1:
            # Single domain with good confidence
            overall_confidence = confidence_scores[detected_domains[0]]
        else:
            # Multiple domains - lower confidence, needs agent planning
            overall_confidence = 0.5  # Medium confidence triggers agent routing
        
        return {
            "domains": detected_domains,
            "confidence": overall_confidence,
            "scores": confidence_scores,
            "complexity": "simple" if len(detected_domains) <= 1 else "complex"
        }
    
    def should_handle_request(self, message: str) -> bool:
        """Router handles all initial requests."""
        return True  # Router is always the entry point
    
    def get_route_destination(self, state: AgentState) -> str:
        """
        Get routing destination for backward compatibility.
        
        This method is kept for compatibility with existing graph structure
        while we transition to the simplified routing.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node destination
        """
        next_action = state.get("next_action")
        target_domain = state.get("target_domain")
        
        if next_action == "escalate":
            return "agent"
        elif next_action == "continue" and target_domain:
            return target_domain
        else:
            return "agent"  # Default fallback 