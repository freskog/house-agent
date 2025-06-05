"""
Agent node for orchestrating multi-domain requests.

This agent uses message-based execution instead of complex state management,
making it much simpler while preserving all functionality.
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from .base_node import BaseNode
from .schemas import AgentState, AgentPlan, create_step_message, aggregate_simple_results
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AgentNode(BaseNode):
    """
    Agent that orchestrates multi-domain requests using message-based execution.
    
    This eliminates the complex state management while preserving all
    multi-step execution capabilities.
    """
    
    def __init__(self, specialists: Dict[str, Any]):
        """
        Initialize simplified agent.
        
        Args:
            specialists: Dictionary of domain -> specialist node (for compatibility)
        """
        super().__init__([], "Agent")
        self.specialists = specialists
        
        # Create LLM for planning (same as before)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            streaming=False
        ).with_structured_output(AgentPlan, method="json_mode")
        
        self.logger.info(f"Agent initialized with {len(specialists)} specialists")
    
    async def handle_request(self, state: AgentState) -> AgentState:
        """
        Handle complex multi-domain requests using simplified message-based execution.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with next step or final response
        """
        user_message = self.get_last_user_message(state)
        if not user_message:
            return self.create_response(
                [AIMessage(content="I'm not sure what you'd like me to help with.")],
                next_action="end"
            )
        
        # Check if this is a fresh request or a continuation
        if self.is_fresh_request(state):
            return await self.handle_fresh_request(user_message.content, state)
        else:
            return await self.handle_continuation(state)
    
    def is_fresh_request(self, state: AgentState) -> bool:
        """
        Determine if this is a fresh user request or a continuation.
        
        Args:
            state: Current state
            
        Returns:
            True if this is a fresh request
        """
        messages = state.get("messages", [])
        
        # If there are no messages or only one human message, it's fresh
        if len(messages) <= 1:
            return True
        
        # Check if the last message is from a specialist (indicating continuation)
        last_message = messages[-1]
        if isinstance(last_message, AIMessage):
            # Look for agent instruction patterns in recent messages
            for msg in reversed(messages[-5:]):  # Check last 5 messages
                if isinstance(msg, HumanMessage) and self.is_agent_instruction(msg):
                    return False  # This is a continuation
        
        return True  # Default to fresh request
    
    async def handle_fresh_request(self, user_request: str, state: AgentState) -> AgentState:
        """
        Handle a fresh user request by creating and starting execution.
        
        Args:
            user_request: Original user request
            state: Current state
            
        Returns:
            Updated state with first step
        """
        # Create execution plan
        plan = await self.create_plan(user_request)
        
        if not plan.steps:
            return self.create_response(
                [AIMessage(content="I couldn't determine how to handle your request.")],
                next_action="end"
            )
        
        # Start execution with first step
        first_step = plan.steps[0]
        step_message = create_step_message(first_step)
        
        return {
            **state,
            "messages": state.get("messages", []) + [step_message],
            "next_action": "continue",
            "target_domain": first_step["domain"]
        }
    
    async def handle_continuation(self, state: AgentState) -> AgentState:
        """
        Handle continuation of multi-step execution.
        
        Args:
            state: Current state with specialist response
            
        Returns:
            Updated state with next step or final response
        """
        messages = state.get("messages", [])
        
        # Extract the original plan from conversation history
        plan = self.extract_plan_from_messages(messages)
        
        if not plan:
            # No plan found, provide final response
            return self.create_final_response(state)
        
        # Determine current step index
        current_step_index = self.get_current_step_index(messages, plan)
        
        # Check if we have more steps
        if current_step_index + 1 < len(plan.steps):
            # Execute next step
            next_step = plan.steps[current_step_index + 1]
            step_message = create_step_message(next_step)
            
            return {
                **state,
                "messages": messages + [step_message],
                "next_action": "continue",
                "target_domain": next_step["domain"]
            }
        else:
            # All steps completed, create final response
            return self.create_final_response(state)
    
    def extract_plan_from_messages(self, messages: List) -> Optional[AgentPlan]:
        """
        Extract the execution plan from message history.
        
        This is a simplified approach - in practice, we could store the plan
        in a special message or use other approaches.
        
        Args:
            messages: Conversation messages
            
        Returns:
            Extracted plan or None
        """
        # For now, recreate a simple plan based on the instructions we see
        steps = []
        current_domain = None
        
        for message in messages:
            if isinstance(message, HumanMessage) and self.is_agent_instruction(message):
                content = message.content
                
                if content.startswith("Search for:"):
                    steps.append({"domain": "search", "action": "lookup", "params": {"query": content[12:]}})
                elif content.startswith("Play:"):
                    steps.append({"domain": "music", "action": "play", "params": {"query": content[5:]}})
                elif content.startswith("Set volume to"):
                    volume = int(content.split()[3].rstrip('%'))
                    steps.append({"domain": "music", "action": "set_volume", "params": {"volume": volume}})
                elif content.startswith("Turn on"):
                    steps.append({"domain": "house", "action": "lights_on", "params": {}})
        
        if steps:
            return AgentPlan(
                steps=steps,
                execution_mode="sequential",
                final_response_template=None
            )
        
        return None
    
    def get_current_step_index(self, messages: List, plan: AgentPlan) -> int:
        """
        Determine the current step index based on completed instructions.
        
        Args:
            messages: Conversation messages
            plan: Execution plan
            
        Returns:
            Current step index (0-based)
        """
        instruction_count = 0
        for message in messages:
            if isinstance(message, HumanMessage) and self.is_agent_instruction(message):
                instruction_count += 1
        
        return instruction_count - 1  # Convert to 0-based index
    
    def create_final_response(self, state: AgentState) -> AgentState:
        """
        Create the final aggregated response.
        
        Args:
            state: Current state
            
        Returns:
            State with final response
        """
        messages = state.get("messages", [])
        user_request = self.get_original_user_request(messages)
        
        # Aggregate results using the simplified approach
        final_response = aggregate_simple_results(messages, user_request)
        
        if final_response:
            return self.create_response(
                [AIMessage(content=final_response)],
                next_action="end"
            )
        else:
            # Silent response (e.g., all music commands)
            return {
                **state,
                "next_action": "end"
            }
    
    def get_original_user_request(self, messages: List) -> str:
        """
        Get the original user request from messages.
        
        Args:
            messages: Conversation messages
            
        Returns:
            Original user request
        """
        for message in messages:
            if isinstance(message, HumanMessage) and not self.is_agent_instruction(message):
                return message.content
        
        return ""
    
    async def create_plan(self, user_request: str) -> AgentPlan:
        """
        Create an execution plan for the request.
        
        Args:
            user_request: User's request
            
        Returns:
            Execution plan
        """
        current_date = datetime.now().strftime("%B %d, %Y")
        
        planning_prompt = f"""You are an AI assistant orchestrator. Today is {current_date}.

User request: "{user_request}"

Available specialist domains:
- music: Handle music playback, volume, Spotify control
- house: Handle home automation, lights, temperature, devices  
- search: Handle web search, information lookup, current events
- clarification: Handle unclear or ambiguous requests

Create an execution plan to fulfill this request. Consider:
1. What domains are needed?
2. What order should they execute in?
3. How should results be combined?

Respond with a JSON object containing:
- steps: List of execution steps, each with domain, action, and params
- execution_mode: "sequential" 
- final_response_template: Optional template for combining results

Available domains: music, house, search, clarification

Example step format:
{{"domain": "search", "action": "lookup", "params": {{"query": "weather today"}}}}

JSON response required."""

        try:
            plan = await self.llm.ainvoke([SystemMessage(content=planning_prompt)])
            return plan
            
        except Exception as e:
            self.logger.error(f"Error creating execution plan: {e}")
            
            # Fallback plan
            if any(keyword in user_request.lower() for keyword in ["search", "find", "weather", "news"]):
                return AgentPlan(
                    steps=[{"domain": "search", "action": "lookup", "params": {"query": user_request}}],
                    execution_mode="sequential"
                )
            else:
                return AgentPlan(
                    steps=[{"domain": "clarification", "action": "clarify", "params": {"request": user_request}}],
                    execution_mode="sequential"
                )
    
    def should_handle_request(self, message: str) -> bool:
        """Agent handles requests when explicitly routed to it."""
        return False  # Agent is only called via routing 