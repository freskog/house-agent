"""
Shared schemas for the house agent system.

This module provides schemas that define the communication contracts between:
- Router Node ↔ Specialist Nodes
- Agent Node ↔ Specialist Nodes
- Specialist Nodes ↔ System

Core Architecture:
- AgentState: Streamlined state
- NodeResponse: Standardized response format
- AgentPlan: Multi-step execution planning
"""

from typing import Dict, Any, Optional, List, Literal
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    """Streamlined state schema with minimal required fields"""
    # Core conversation (only required field)
    messages: Annotated[List[SystemMessage | HumanMessage | AIMessage | ToolMessage], add_messages]
    
    # Simple routing hints (set by nodes, used by graph)
    next_action: Optional[Literal["end", "escalate", "continue"]]
    target_domain: Optional[str]
    
    # Audio/client context (unchanged for compatibility)
    audio_server: Optional[Any]
    current_client: Optional[Any]

class NodeResponse(BaseModel):
    """Standardized response from any node"""
    messages: List[AIMessage] = Field(description="Messages to add to conversation")
    next_action: Literal["end", "escalate", "continue"] = Field(description="What to do next")
    target_domain: Optional[str] = Field(default=None, description="Which domain to route to if continuing")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context if needed")

def route_next_node(state: AgentState) -> str:
    """
    Unified routing function for all nodes.
    
    This single function replaces all the complex routing logic
    scattered throughout the old system.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name or END
    """
    next_action = state.get("next_action")
    target_domain = state.get("target_domain")
    
    if next_action == "end":
        return "END"
    elif next_action == "escalate":
        return "agent"
    elif next_action == "continue" and target_domain:
        # Route to the specified domain
        if target_domain in ["music", "house", "search", "clarification"]:
            return target_domain
        else:
            return "agent"  # Default to agent for unknown domains
    else:
        return "END"  # Default to end if no clear action

def create_step_message(step: Dict[str, Any]) -> HumanMessage:
    """
    Create a message that represents an agent step for a specialist.
    
    Args:
        step: Step dictionary with domain, action, and params
        
    Returns:
        HumanMessage that specialists can understand
    """
    action = step.get("action", "execute")
    params = step.get("params", {})
    
    if action == "lookup" and "query" in params:
        # Search-specific formatting
        return HumanMessage(content=f"Search for: {params['query']}")
    elif action == "play" and "query" in params:
        # Music-specific formatting  
        return HumanMessage(content=f"Play: {params['query']}")
    elif action == "set_volume" and "volume" in params:
        # Volume control formatting
        return HumanMessage(content=f"Set volume to {params['volume']}%")
    elif action == "lights_on" and "room" in params:
        # Light control formatting
        return HumanMessage(content=f"Turn on lights in {params['room']}")
    else:
        # Generic formatting
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return HumanMessage(content=f"Execute {action}" + (f" with {param_str}" if param_str else ""))

def extract_results_from_messages(messages: List) -> Dict[str, str]:
    """
    Extract specialist results from conversation messages.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Dictionary mapping domains to their responses
    """
    results = {}
    current_domain = None
    
    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage) and message.content.startswith(("Execute", "Search for:", "Play:", "Set volume", "Turn on")):
            # This is an agent instruction - determine domain from next AI response
            if i + 1 < len(messages) and isinstance(messages[i + 1], AIMessage):
                response_content = messages[i + 1].content
                
                # Determine domain from instruction content
                if message.content.startswith(("Play:", "Set volume")):
                    current_domain = "music"
                elif message.content.startswith("Search for:"):
                    current_domain = "search"
                elif message.content.startswith("Turn on"):
                    current_domain = "house"
                else:
                    current_domain = "unknown"
                
                results[current_domain] = response_content
    
    return results

def aggregate_simple_results(messages: List, original_request: str) -> str:
    """
    Aggregate results from messages into a final response.
    
    Args:
        messages: Conversation messages
        original_request: Original user request
        
    Returns:
        Final aggregated response
    """
    results = extract_results_from_messages(messages)
    
    if not results:
        return "I completed your request."
    
    # Filter out silent responses (empty or whitespace)
    meaningful_results = {domain: response for domain, response in results.items() 
                         if response and response.strip()}
    
    if not meaningful_results:
        return ""  # Silent response for all-silent operations
    
    if len(meaningful_results) == 1:
        return list(meaningful_results.values())[0]
    else:
        # Multiple domain results
        response_parts = []
        for domain, response in meaningful_results.items():
            if domain.lower() not in response.lower():
                response_parts.append(f"{domain.title()}: {response}")
            else:
                response_parts.append(response)
        
        return "I've completed your request:\n\n" + "\n".join(response_parts)

class AgentPlan(BaseModel):
    """Agent planning schema for multi-domain requests."""
    steps: List[Dict[str, Any]] = Field(
        description="Ordered list of steps to execute"
    )
    execution_mode: Literal["sequential", "parallel"] = Field(
        description="Whether steps should run sequentially or in parallel"
    )
    final_response_template: Optional[str] = Field(
        None,
        description="Template for aggregating results into final response"
    )

class ToolCall(BaseModel):
    """Schema for individual tool calls."""
    name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(description="Arguments to pass to the tool")

class SpecialistResponse(BaseModel):
    """Response schema for specialist nodes (legacy)."""
    action: Literal["execute_tools", "respond", "escalate"] = Field(
        description="Action the specialist will take"
    )
    
    # For execute_tools path
    tool_calls: Optional[List[ToolCall]] = Field(
        None,
        description="Tools to execute if action is execute_tools"
    )
    needs_tool_results: bool = Field(
        False, 
        description="Whether LLM needs to see tool results (false = fast path)"
    )
    success_response: Optional[str] = Field(
        None,
        description="Pre-computed response for successful tool execution"
    )
    failure_response: Optional[str] = Field(
        None,
        description="Pre-computed response for failed tool execution"
    )
    
    # For respond path (direct response, no tools)
    response: Optional[str] = Field(
        None,
        description="Direct response text if action is respond"
    )
    
    # For escalate path (only if called_by != "agent")
    escalation_reason: Optional[str] = Field(
        None,
        description="Reason for escalating to agent"
    )
    detected_domains: Optional[List[str]] = Field(
        None,
        description="List of domains detected in the request"
    )

 