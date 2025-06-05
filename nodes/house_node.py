"""
House automation node for handling home device control and automation requests.

This node manages all house automation functionality including:
- MCP client setup and connection management
- Device control (lights, temperature, switches, etc.)
- Home automation workflows
- Home Assistant prompt loading and context
- Device state management
- Structured outputs and escalation for hierarchical routing

Currently integrates with Home Assistant but designed to support multiple home automation platforms.
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
from .base_node import BaseNode, AgentState
from .schemas import SpecialistResponse

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class HouseNode(BaseNode):
    """Node for handling house device control and automation requests with structured outputs."""
    
    def __init__(self):
        """Initialize House node with lazy MCP setup and structured LLM."""
        self.ha_api_key = os.getenv("HA_API_KEY")
        self.ha_url = os.getenv("HA_URL", "http://10.10.100.126:8123/mcp_server/sse")
        self.mcp_client = None
        self.homeassistant_content = None
        self._initialization_attempted = False
        
        # Start with no tools - they'll be loaded lazily
        super().__init__([], "House")
        
        # Initialize LLM for structured outputs
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            streaming=False
        ).with_structured_output(SpecialistResponse, method="function_calling")
        
        if not self.ha_api_key:
            self.logger.warning("No HA_API_KEY found - House automation functionality will not be available")
        else:
            self.logger.info("House node ready (tools will be loaded on first use)")
    
    async def initialize_mcp_client(self) -> bool:
        """
        Initialize the MCP client for Home Assistant.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if not self.ha_api_key:
            self.logger.warning("Cannot initialize MCP client without HA_API_KEY")
            return False
        
        try:
            # Set up SSE connection configuration for Home Assistant
            ha_connection: SSEConnection = {
                "transport": "sse",
                "url": self.ha_url,
                "headers": {"Authorization": f"Bearer {self.ha_api_key}"},
                "timeout": 10,             # HTTP connection timeout
                "sse_read_timeout": 60,    # SSE read timeout
                "session_kwargs": {}       # Additional session parameters if needed
            }
            
            # Create MCP client with Home Assistant connection
            connections = {"home_assistant": ha_connection}
            self.mcp_client = MultiServerMCPClient(connections=connections)
            
            self.logger.info("Successfully initialized Home Assistant MCP client")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Home Assistant MCP client: {e}")
            return False
    
    async def load_tools(self) -> List:
        """
        Load tools from the MCP client.
        
        Returns:
            List of Home Assistant tools
        """
        if not self.mcp_client:
            await self.initialize_mcp_client()
        
        if not self.mcp_client:
            self.logger.warning("No MCP client available for loading tools")
            return []
        
        try:
            self.logger.info("Getting tools from Home Assistant MCP client...")
            
            # Handle different MCP client implementations
            try:
                mcp_tools = await self.mcp_client.get_tools()
            except TypeError as e:
                # Handle the case where get_tools() returns a list directly
                if "object list can't be used in 'await' expression" in str(e) or "can't be used in 'await' expression" in str(e):
                    self.logger.warning("get_tools() returned a list directly, calling without await")
                    mcp_tools = self.mcp_client.get_tools()
                else:
                    raise e
            
            # Update our tools
            self.tools = mcp_tools
            
            self.logger.info(f"Loaded {len(mcp_tools)} tools from Home Assistant MCP client")
            return mcp_tools
            
        except Exception as e:
            self.logger.error(f"Error loading Home Assistant MCP tools: {e}")
            return []
    
    async def load_homeassistant_prompt(self) -> Optional[str]:
        """
        Load prompt content from Home Assistant MCP server.
        
        Returns:
            Prompt content string or None if not available
        """
        if not self.mcp_client:
            self.logger.warning("No MCP client available for loading prompts")
            return None
        
        try:
            self.logger.info("Loading prompts from Home Assistant...")
            
            # Check if the session method exists
            if not hasattr(self.mcp_client, 'session'):
                self.logger.warning("MultiServerMCPClient does not have 'session' method - skipping prompts")
                return None
            
            async with self.mcp_client.session("home_assistant") as session:
                # Handle different MCP implementations for prompts
                try:
                    prompts = await session.list_prompts()
                except TypeError as e:
                    # If list_prompts() is not awaitable, call it directly
                    if "can't be used in 'await' expression" in str(e):
                        prompts = session.list_prompts()
                    else:
                        raise e
                
                # Handle different return types
                prompts_list = prompts.prompts if hasattr(prompts, "prompts") else prompts
                
                if prompts_list:
                    self.logger.info(f"Found {len(prompts_list)} prompts:")
                    for i, prompt in enumerate(prompts_list):
                        prompt_name = prompt.name if hasattr(prompt, "name") else prompt.get("name", "Unknown")
                        self.logger.info(f"{i+1}. {prompt_name}")
                        
                    # Try to get the first prompt
                    first_prompt = prompts_list[0]
                    prompt_name = first_prompt.name if hasattr(first_prompt, "name") else first_prompt.get("name", "Unknown")
                    self.logger.info(f"Getting content for prompt '{prompt_name}'...")
                    
                    try:
                        prompt_details = await session.get_prompt(prompt_name)
                        content = None
                        
                        # Handle different MCP prompt response formats
                        if hasattr(prompt_details, "content"):
                            content = prompt_details.content
                        elif isinstance(prompt_details, dict):
                            content = prompt_details.get("content")
                        elif hasattr(prompt_details, "messages") and prompt_details.messages:
                            # Handle GetPromptResult format with messages
                            first_message = prompt_details.messages[0]
                            if hasattr(first_message, "content"):
                                if hasattr(first_message.content, "text"):
                                    content = first_message.content.text
                                else:
                                    content = str(first_message.content)
                        
                        if not content:
                            # Last resort - try to extract text from the object
                            content_str = str(prompt_details)
                            if "Static Context:" in content_str:
                                # Extract the part after "Static Context:"
                                content = content_str.split("Static Context:", 1)[1].strip()
                                # Clean up formatting
                                content = content.replace("\\n", "\n").replace("\\'", "'")
                                if content.endswith("')]"):
                                    content = content[:-3]
                        
                        if content:
                            self.homeassistant_content = content
                            self.logger.info(f"Successfully loaded Home Assistant prompt content ({len(content)} chars)")
                            return content
                        else:
                            self.logger.warning("No content found in prompt details")
                            return None
                            
                    except Exception as e:
                        self.logger.error(f"Failed to get prompt details: {e}")
                        return None
                else:
                    self.logger.info("No prompts found in Home Assistant")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to load Home Assistant prompts: {e}")
            return None
    
    def should_handle_request(self, message: str) -> bool:
        """
        Determine if this node should handle the home automation request.
        
        Args:
            message: User message to evaluate
            
        Returns:
            True if this node should handle the request, False otherwise
        """
        message_lower = message.lower()
        
        # Home automation keywords
        home_keywords = [
            "lights", "light", "lamp", "temperature", "temp", "heating", "cooling",
            "turn on", "turn off", "switch on", "switch off", "dim", "bright",
            "thermostat", "ac", "air conditioning", "fan", "blinds", "curtains",
            "door", "lock", "unlock", "secure", "alarm", "sensor", "device"
        ]
        
        return any(keyword in message_lower for keyword in home_keywords)
    
    async def handle_request(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle home automation requests using structured outputs and three-path routing.
        
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
                AIMessage(content="I didn't receive a clear home automation request.")
            ])
        
        # Ensure tools are loaded
        await self._ensure_tools_loaded()
        
        if not self.tools:
            return self.create_response([
                AIMessage(content="Home automation is not available right now. Please check your Home Assistant connection.")
            ])
        
        # Use structured output to decide action
        specialist_response = await self._get_structured_response(user_message.content, called_by, state)
        
        # Handle based on action
        if specialist_response.action == "escalate":
            return await self._handle_escalation(specialist_response, state, user_message.content)
        elif specialist_response.action == "respond":
            return self.create_response([AIMessage(content=specialist_response.response)])
        elif specialist_response.action == "execute_tools":
            return await self._execute_tools(specialist_response, state)
        else:
            self.logger.warning(f"Unknown action: {specialist_response.action}")
            return self.create_response([AIMessage(content="I'm not sure how to handle that home automation request.")])
    
    async def _get_structured_response(self, user_request: str, called_by: str, state: AgentState) -> SpecialistResponse:
        """Get structured response from LLM to determine action path."""
        current_date = datetime.now().strftime("%B %d, %Y")
        ha_context = self.get_homeassistant_context()
        
        prompt = f"""You are a home automation specialist. Today is {current_date}.

User request: "{user_request}"
Called by: {called_by}

{ha_context}

IMPORTANT ENTITY MATCHING INSTRUCTIONS:
When the user asks to control devices, you MUST match their request to the exact entity names from the Home Assistant context above.

Entity Matching Rules:
1. "lights in office" / "office lights" → Look for entities with "office" in the name
2. "living room lights" → Look for entities with "living room" in the name  
3. "bedroom temperature" → Look for thermostats/sensors with "bedroom" in the name
4. "front door lock" → Look for locks with "front door" in the name

Available Tools:
- HassTurnOn(name="exact_entity_name") - Turn on a device
- HassTurnOff(name="exact_entity_name") - Turn off a device  
- HassSetTemperature(name="exact_entity_name", temperature=X) - Set temperature
- (Check context for full list of available tools)

TOOL_CALLS FORMAT REQUIREMENTS:
For execute_tools action, tool_calls MUST be a list of objects with "name" and "arguments" keys:
[{{"name": "HassTurnOn", "arguments": {{"name": "Fredrik's office ceiling light"}}}}]

Domain Detection Rules:
- If request involves ONLY home automation (lights, temperature, etc.) → this is purely house domain
- If request involves home automation AND other domains (music, search, etc.) → escalate if called_by="router"
- If called_by="agent" → never escalate, just handle house automation part

Choose ONE action:

1. execute_tools: For home automation requests needing device control
   - STEP 1: Find matching entity name in Home Assistant context above
   - STEP 2: Generate tool_calls with exact entity name
   - STEP 3: Set needs_tool_results=false for simple commands
   - STEP 4: Provide success_response and failure_response

2. respond: For general information about home automation or explanations

3. escalate: ONLY if called_by="router" AND request involves multiple domains

EXAMPLE FOR "turn on lights in office":
{{
  "action": "execute_tools",
  "tool_calls": [{{"name": "HassTurnOn", "arguments": {{"name": "Fredrik's office ceiling light"}}}}],
  "needs_tool_results": false,
  "success_response": "I've turned on the office lights.",
  "failure_response": "I couldn't turn on the office lights."
}}

CRITICAL: Use EXACT entity names from the Home Assistant context!

Respond with structured output following SpecialistResponse schema."""

        try:
            response = await self.llm.ainvoke([SystemMessage(content=prompt)])
            return response
        except Exception as e:
            self.logger.error(f"Error getting structured response: {e}")
            # Fallback to direct response
            return SpecialistResponse(
                action="respond",
                response="I encountered an issue processing your home automation request."
            )
    
    async def _handle_escalation(self, specialist_response, state: AgentState, user_request: str) -> Dict[str, Any]:
        """Handle escalation to agent for multi-domain requests."""
        return self.create_escalation_response(
            reason=specialist_response.escalation_reason or "Multi-domain request detected",
            domains=specialist_response.detected_domains or ["house"],
            original_request=user_request
        )

    
    async def _execute_tools(self, specialist_response: SpecialistResponse, state: AgentState) -> Dict[str, Any]:
        """Execute tools based on specialist response."""
        if not specialist_response.tool_calls:
            return self.create_response([
                AIMessage(content=specialist_response.failure_response or "No tools were specified to execute.")
            ])
        
        # If we don't need tool results, use fast path with pre-computed responses
        if not specialist_response.needs_tool_results:
            try:
                # Execute tools in background for side effects
                tool_results = await self._execute_tool_calls(specialist_response.tool_calls, state)
                
                # Check if any tool failed
                has_failures = any(
                    result.get("error") or self._has_tool_failure(result.get("content", ""))
                    for result in tool_results
                )
                
                if has_failures:
                    response_text = specialist_response.failure_response or "Home automation command failed."
                else:
                    response_text = specialist_response.success_response or ""
                
                return self.create_response([AIMessage(content=response_text)])
                
            except Exception as e:
                self.logger.error(f"Tool execution failed: {e}")
                return self.create_response([
                    AIMessage(content=specialist_response.failure_response or "Home automation command failed.")
                ])
        else:
            # Full path - execute tools and process results
            try:
                tool_results = await self._execute_tool_calls(specialist_response.tool_calls, state)
                
                # Get user request from state for context
                user_request = ""
                if state.get("messages"):
                    last_human_msg = next((msg for msg in reversed(state["messages"]) if hasattr(msg, 'type') and msg.type == 'human'), None)
                    if last_human_msg:
                        user_request = getattr(last_human_msg, 'content', '')
                
                # Process tool results and create response
                response_text = await self._process_tool_results(tool_results, user_request)
                return self.create_response([AIMessage(content=response_text)])
                
            except Exception as e:
                self.logger.error(f"Tool execution failed: {e}")
                return self.create_response([
                    AIMessage(content="I encountered an error executing your home automation request.")
                ])
    
    async def _execute_tool_calls(self, tool_calls, state: AgentState) -> List[Dict[str, Any]]:
        """Execute the specified tool calls."""
        from nodes.schemas import ToolCall
        
        results = []
        
        for tool_call in tool_calls:
            # Handle both ToolCall objects and dictionaries for backward compatibility
            if isinstance(tool_call, ToolCall):
                tool_name = tool_call.name
                tool_args = tool_call.arguments
            else:
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("arguments", {})
            
            # Find the tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                results.append({
                    "error": f"Tool {tool_name} not found",
                    "content": f"Error: Tool {tool_name} is not available"
                })
                continue
            
            try:
                # Execute the tool
                result = await tool.ainvoke(tool_args) if hasattr(tool, 'ainvoke') else tool.invoke(tool_args)
                results.append({
                    "tool_name": tool_name,
                    "content": result
                })
            except Exception as e:
                self.logger.error(f"Error executing tool {tool_name}: {e}")
                results.append({
                    "error": str(e),
                    "content": f"Error executing {tool_name}: {str(e)}"
                })
        
        return results
    
    def _has_tool_failure(self, content: str) -> bool:
        """Check if Home Assistant tool response indicates failure."""
        if not content:
            return False
        
        try:
            import json
            # Try to parse as JSON to check the actual failed array
            if isinstance(content, str) and content.strip().startswith('{"'):
                response_data = json.loads(content)
                if isinstance(response_data, dict) and "data" in response_data:
                    data = response_data["data"]
                    # Check if there are actual failed items
                    failed_items = data.get("failed", [])
                    return len(failed_items) > 0
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback to string-based detection if JSON parsing fails
            pass
        
        # Fallback: look for error indicators in the content
        content_lower = str(content).lower()
        error_indicators = ["error", "exception", "could not", "unable to", "not found"]
        return any(indicator in content_lower for indicator in error_indicators)
    
    async def _process_tool_results(self, tool_results: List[Dict[str, Any]], user_request: str = "") -> str:
        """Process tool results into a human-readable response."""
        if not tool_results:
            return "No results from home automation tools."
        
        responses = []
        raw_content = []
        has_informational_content = False
        
        for result in tool_results:
            content = result.get("content", "")
            error = result.get("error")
            
            if error:
                responses.append(f"Error: {error}")
            elif content:
                # Check if this is a Home Assistant device status response
                processed_response = self._interpret_ha_response(content)
                if processed_response:
                    responses.append(processed_response)
                    has_informational_content = True
                else:
                    # Check if it's a long informational response that should be summarized
                    if len(str(content)) > 100 and self._is_informational_response(content):
                        raw_content.append(str(content))
                        has_informational_content = True
                    else:
                        responses.append(content)
        
        # If we have informational content and it's a query (not an action), use LLM summarization
        if has_informational_content and raw_content and self._is_informational_query(user_request):
            try:
                summary = await self._create_ha_summary(raw_content, user_request)
                if summary:
                    responses.append(summary)
            except Exception as e:
                self.logger.error(f"Error creating HA summary: {e}")
                # Fallback to original responses
                pass
        
        if responses:
            return " ".join(responses)
        else:
            return ""  # Silent response for successful action commands

    def _interpret_ha_response(self, content: str) -> str:
        """
        Interpret Home Assistant tool responses and convert to human-readable format.
        
        Args:
            content: Raw tool response content
            
        Returns:
            Human-readable interpretation or empty string if not interpretable
        """
        if not content:
            return ""
        
        try:
            import json
            
            # Check if it's a JSON response with device status
            if isinstance(content, str) and content.strip().startswith('{"') and '"result"' in content:
                json_data = json.loads(content)
                
                if json_data.get("success") and "result" in json_data:
                    result = json_data["result"]
                    
                    # Check if it's a device overview/status response
                    if "Live Context: An overview" in result or "devices in this smart home" in result:
                        return self._parse_device_status_response(result)
                    
            # Check for simple device control responses
            if "turned on" in content.lower() or "turned off" in content.lower():
                return content
                
                        # For other responses, try to extract meaningful information
            if any(keyword in content.lower() for keyword in ["success", "completed", "failed", "error"]):
                return content
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.debug(f"Could not parse HA response as JSON: {e}")
        
        return ""  # Return empty if we can't interpret it
    
    def _is_informational_query(self, user_request: str) -> bool:
        """Check if the user request is asking for information rather than performing an action."""
        if not user_request:
            return False
        
        query_lower = user_request.lower()
        
        # Check for question words and patterns
        info_patterns = [
            "what", "how", "where", "when", "which", "who", "is", "are", "do", "does",
            "status", "state", "check", "show", "tell me", "what's", "how's", "are the",
            "is the", "what are", "how are", "which are"
        ]
        
        # Action patterns that should remain silent
        action_patterns = [
            "turn on", "turn off", "switch on", "switch off", "set", "dim", "brighten",
            "lock", "unlock", "open", "close", "start", "stop", "pause", "play"
        ]
        
        # If it contains action patterns, it's not informational
        if any(pattern in query_lower for pattern in action_patterns):
            return False
        
        # If it contains info patterns, it's informational
        return any(pattern in query_lower for pattern in info_patterns)
    
    def _is_informational_response(self, content: str) -> bool:
        """Check if content appears to be informational (status/overview) rather than action confirmation."""
        if not content:
            return False
        
        content_str = str(content).lower()
        
        # Signs of informational content
        info_indicators = [
            "overview", "context", "devices", "state:", "status", "temperature:",
            "light:", "sensor:", "available", "current", "smart home"
        ]
        
        return any(indicator in content_str for indicator in info_indicators)
    
    async def _create_ha_summary(self, raw_content: List[str], user_request: str) -> str:
        """Create a human-readable summary of Home Assistant responses using LLM."""
        if not raw_content:
            return ""
        
        combined_content = "\n\n".join(raw_content)
        
        system_prompt = f"""You are summarizing Home Assistant device information for a voice assistant.

User asked: "{user_request}"

Home Assistant data:
{combined_content[:1000]}

Create a concise, natural response that:
- Directly answers the user's question about their smart home
- Is easy to understand when spoken aloud
- Focuses on the most relevant device states/information
- Keeps it under 2-3 sentences
- Sounds conversational and natural

If the user asked about specific devices (lights, temperature, etc.), focus on those.

Response:"""

        try:
            summary_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=False, max_tokens=100)
            messages = [SystemMessage(content=system_prompt)]
            response = await summary_llm.ainvoke(messages)
            
            if response.content:
                return response.content.strip()
            else:
                return ""
                
        except Exception as e:
            self.logger.error(f"Error in HA LLM summarization: {e}")
            return ""
    
    def _parse_device_status_response(self, result: str) -> str:
        """
        Parse a Home Assistant device status overview and extract relevant information.
        
        Args:
            result: The result string containing device information
            
        Returns:
            Human-readable summary of device states
        """
        if not result:
            return "No device information available."
        
        try:
            # Look for office lights specifically in the result
            if "Fredrik's office ceiling light" in result:
                # Extract the state of Fredrik's office light
                lines = result.split('\n')
                for i, line in enumerate(lines):
                    if "Fredrik's office ceiling light" in line:
                        # Look for state in the next few lines
                        for j in range(i, min(i + 5, len(lines))):
                            check_line = lines[j]
                            if 'state: ' in check_line:
                                state_val = check_line.split('state: ')[1].strip().strip("'\"")
                                if state_val == 'on':
                                    return "Yes, the office ceiling light is on."
                                elif state_val == 'off':
                                    return "No, the office ceiling light is off."
                                elif state_val == 'unavailable':
                                    return "The office ceiling light is currently unavailable."
                                break
                        break
            
            # If we can't find specific office light info, provide a general response
            if 'light' in result:
                return "Light information is available in the smart home system."
            else:
                return "Device information retrieved successfully."
                
        except Exception as e:
            self.logger.error(f"Error parsing device status response: {e}")
            return "Error interpreting device status information."
    
    async def _ensure_tools_loaded(self):
        """Ensure tools are loaded (lazy loading)."""
        if not self.tools and not self._initialization_attempted:
            self._initialization_attempted = True
            self.logger.info("Loading Home Assistant tools (lazy initialization)...")
            await self.load_tools()
            await self.load_homeassistant_prompt()
            if self.tools:
                self.logger.info(f"Successfully loaded {len(self.tools)} Home Assistant tools")
            else:
                self.logger.warning("No Home Assistant tools loaded - check configuration")
    
    def get_available_devices(self) -> List[str]:
        """
        Get list of available devices from loaded tools.
        
        Returns:
            List of available device names/types
        """
        if not self.tools:
            return []
        
        devices = []
        for tool in self.tools:
            # Extract device info from tool names/descriptions
            tool_name = getattr(tool, 'name', 'Unknown')
            devices.append(tool_name)
        
        return devices
    
    def get_homeassistant_context(self) -> str:
        """
        Get Home Assistant context information for prompts.
        
        Returns:
            Context string with device information and capabilities
        """
        if self.homeassistant_content:
            return f"Home Assistant Context:\n{self.homeassistant_content}"
        else:
            devices = self.get_available_devices()
            if devices:
                return f"Available Home Assistant devices/tools: {', '.join(devices[:10])}{'...' if len(devices) > 10 else ''}"
            else:
                return "Home Assistant integration available but no devices currently loaded."
    
    def get_home_keywords(self) -> List[str]:
        """Get list of keywords that indicate home automation requests."""
        return [
            "lights", "light", "lamp", "temperature", "temp", "heating", "cooling",
            "turn on", "turn off", "switch on", "switch off", "dim", "bright",
            "thermostat", "ac", "air conditioning", "fan", "blinds", "curtains",
            "door", "lock", "unlock", "secure", "alarm", "sensor", "device",
            "brightness", "scene", "automation", "schedule", "timer"
        ] 