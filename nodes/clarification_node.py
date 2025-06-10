"""
Clarification node for handling unclear or ambiguous requests.

This node manages requests that need clarification including:
- Vague or unclear user inputs
- Ambiguous commands that could apply to multiple domains
- Requests that lack sufficient context
- Error recovery and guidance
- Structured outputs for hierarchical routing

Provides helpful guidance to users when their intent is unclear.
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from .base_node import BaseNode, AgentState
from .schemas import SpecialistResponse

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ClarificationNode(BaseNode):
    """Node for handling clarification and ambiguous requests with structured outputs."""
    
    def __init__(self):
        """Initialize Clarification node with structured LLM (no tools needed)."""
        # No tools needed for clarification - just LLM responses
        super().__init__([], "Clarification")
        
        # Initialize LLM for structured outputs
        self.llm = ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0,
            streaming=True
        ).with_structured_output(SpecialistResponse)
        
        self.logger.info("Initialized Clarification node with structured outputs")
    
    def should_handle_request(self, message: str) -> bool:
        """
        Determine if this node should handle the clarification request.
        
        Args:
            message: User message to evaluate
            
        Returns:
            True if this node should handle the request, False otherwise
        """
        message_lower = message.lower().strip()
        
        # Unclear input patterns
        unclear_patterns = [
            "um", "uh", "er", "hmm", "sorry", "unclear", "what", "huh", "repeat",
            "i don't know", "not sure", "maybe", "help", "confused"
        ]
        
        # Very short messages that might be unclear
        is_very_short = len(message.strip()) < 3
        
        # Check for unclear patterns
        has_unclear_pattern = any(pattern in message_lower for pattern in unclear_patterns)
        
        return has_unclear_pattern or is_very_short
    
    async def handle_request(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle clarification requests using structured outputs.
        
        Args:
            state: Current agent state
            config: Optional configuration
            
        Returns:
            Updated state after handling the request
        """
        # Get context about who called us
        called_by = state.get("called_by", "router")
        user_message = self.get_last_user_message(state)
        
        if not user_message:
            return self.create_response([
                AIMessage(content="I'm not sure what you'd like me to help with. Could you please tell me what you need?")
            ])
        
        # Use structured output to decide action
        specialist_response = await self._get_structured_response(user_message.content, called_by, state)
        
        # Handle based on action
        if specialist_response.action == "escalate":
            return await self._handle_escalation(specialist_response, state, user_message.content)
        elif specialist_response.action == "respond":
            return self.create_response([AIMessage(content=specialist_response.response)])
        elif specialist_response.action == "execute_tools":
            # Clarification node doesn't execute tools, just respond
            return self.create_response([AIMessage(content=specialist_response.response or "I'm not sure how to help with that.")])
        else:
            self.logger.warning(f"Unknown action: {specialist_response.action}")
            return self.create_response([AIMessage(content="I'm not sure how to help with that. Could you please be more specific?")])
    
    async def _get_structured_response(self, user_request: str, called_by: str, state: AgentState) -> SpecialistResponse:
        """Get structured response from LLM to determine action path."""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        prompt = f"""You are a clarification specialist. Today is {current_date}.

User request: "{user_request}"
Called by: {called_by}

Your role is to help with unclear, ambiguous, or vague requests. You can:
- Ask for clarification when user intent is unclear
- Provide helpful guidance about what I can do
- Suggest specific actions when requests are too vague
- Handle error recovery when users are confused

Available domains to guide users toward:
- music: Playing music, volume control, Spotify commands
- house: Home automation, lights, temperature, devices
- search: Web search, information lookup, current events, weather

Domain Detection Rules:
- Clarification is usually a single-domain response (just helping the user)
- If the unclear request seems to involve multiple domains, you can escalate if called_by="router"
- If called_by="agent" → never escalate, just provide clarification

Choose ONE action:

1. respond: For providing clarification, guidance, or asking follow-up questions
   - Most common action for this specialist
   - Help users understand what they can ask for
   - Ask specific clarifying questions

2. escalate: ONLY if called_by="router" AND the unclear request hints at multiple domains
   - Rare case - usually clarification is single-domain
   - Only if you detect the user might need multi-domain help

Examples:
- "um" → respond (ask what they need help with)
- "help" → respond (explain what I can do)
- "what can you do" → respond (list capabilities)
- "uh, play music and lights" → escalate if called_by="router" (multi-domain)

For unclear inputs, be helpful and specific about what the user can ask for.

Respond with structured output following SpecialistResponse schema."""

        try:
            response = await self.llm.ainvoke([SystemMessage(content=prompt)])
            return response
        except Exception as e:
            self.logger.error(f"Error getting structured response: {e}")
            # Fallback to direct response
            return SpecialistResponse(
                action="respond",
                response="I'm not sure what you'd like me to help with. Could you please be more specific about what you need?"
            )
    
    async def _handle_escalation(self, specialist_response, state: AgentState, user_request: str) -> Dict[str, Any]:
        """Handle escalation to agent for multi-domain requests."""
        return self.create_escalation_response(
            reason=specialist_response.escalation_reason or "Multi-domain clarification needed",
            domains=specialist_response.detected_domains or ["clarification"],
            original_request=user_request
        )

    
    def get_clarification_keywords(self) -> List[str]:
        """Get list of keywords that indicate clarification needs."""
        return [
            "um", "uh", "er", "hmm", "sorry", "unclear", "what", "huh", "repeat",
            "i don't know", "not sure", "maybe", "help", "confused", "explain",
            "how", "what can you do", "capabilities", "commands", "options"
        ]
    
    def create_helpful_response(self, context: str = "") -> str:
        """Create a helpful response explaining what the system can do."""
        base_response = """I can help you with several things:

• Music: Play songs, control volume, pause/skip tracks, create radio stations
• Home: Control lights, temperature, and other smart home devices  
• Information: Search the web, check weather, get news and current events

What would you like me to help you with?"""
        
        if context:
            return f"{context}\n\n{base_response}"
        else:
            return base_response 