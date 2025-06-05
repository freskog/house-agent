"""
Home Assistant node for handling home automation and device control requests.

This node manages all Home Assistant-related functionality including:
- MCP client setup and connection management
- Device control (lights, temperature, switches, etc.)
- Home automation workflows
- HA-specific prompt loading and context
- Device state management
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
from .base_node import BaseNode, AgentState
import os
import logging

logger = logging.getLogger(__name__)

class HomeAssistantNode(BaseNode):
    """Node for handling Home Assistant device control and automation requests."""
    
    def __init__(self):
        """Initialize Home Assistant node with lazy MCP setup."""
        self.ha_api_key = os.getenv("HA_API_KEY")
        self.ha_url = "http://10.10.100.126:8123/mcp_server/sse"
        self.mcp_client = None
        self.homeassistant_content = None
        self._initialization_attempted = False
        
        # Start with no tools - they'll be loaded lazily
        super().__init__([], "HomeAssistant")
        
        if not self.ha_api_key:
            self.logger.warning("No HA_API_KEY found - Home Assistant functionality will not be available")
        else:
            self.logger.info("Home Assistant node ready (tools will be loaded on first use)")
    
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
            if self.tools:
                self.tool_node = self.__class__.__bases__[0](self.tools, self.node_name).__dict__['tool_node'].__class__(self.tools)
            
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
                        if hasattr(prompt_details, "content"):
                            content = prompt_details.content
                        elif isinstance(prompt_details, dict):
                            content = prompt_details.get("content")
                        else:
                            content = None
                        
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
            "home", "house", "room", "bedroom", "kitchen", "living room",
            "thermostat", "climate", "hvac", "fan", "door", "window", "garage",
            "security", "alarm", "lock", "unlock", "sensor", "motion", "camera"
        ]
        
        return any(keyword in message_lower for keyword in home_keywords)
    
    async def handle_request(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle Home Assistant-related requests with lazy tool loading.
        
        Args:
            state: Current agent state
            config: Optional configuration
            
        Returns:
            Updated state after handling the request
        """
        # Lazy load tools on first HA request
        await self._ensure_tools_loaded()
        
        if not self.tools and self.ha_api_key:
            self.logger.warning("Home Assistant tools failed to load")
            return self.create_response([
                AIMessage(content="Sorry, I'm having trouble connecting to Home Assistant right now.")
            ])
        elif not self.ha_api_key:
            return self.create_response([
                AIMessage(content="Sorry, Home Assistant is not configured.")
            ])
        
        # Set current domain in state
        state = self.update_state(state, {"current_domain": "home_assistant"})
        
        # The actual tool calling will be handled by the LLM in the main agent
        # This node primarily provides the tools and handles post-execution logic
        return state
    
    async def _ensure_tools_loaded(self):
        """Ensure HA tools are loaded, attempting lazy initialization if needed."""
        if self.tools or self._initialization_attempted:
            return  # Already loaded or already tried
        
        if not self.ha_api_key:
            self._initialization_attempted = True
            return
        
        self.logger.info("Lazy loading Home Assistant tools...")
        try:
            # Try to load tools
            await self.load_tools()
            await self.load_homeassistant_prompt()
            self.logger.info(f"Successfully lazy-loaded {len(self.tools)} Home Assistant tools")
        except Exception as e:
            self.logger.error(f"Failed to lazy-load Home Assistant tools: {e}")
        finally:
            self._initialization_attempted = True
    
    def get_available_devices(self) -> List[str]:
        """
        Get list of available Home Assistant devices from tools.
        
        Returns:
            List of device names/types that can be controlled
        """
        if not self.tools:
            return []
        
        # Extract device types from tool names/descriptions
        device_types = set()
        for tool in self.tools:
            tool_name = getattr(tool, 'name', '').lower()
            if 'light' in tool_name:
                device_types.add('lights')
            elif 'climate' in tool_name or 'temperature' in tool_name:
                device_types.add('climate control')
            elif 'switch' in tool_name:
                device_types.add('switches')
            elif 'sensor' in tool_name:
                device_types.add('sensors')
        
        return list(device_types)
    
    def get_homeassistant_context(self) -> str:
        """
        Get Home Assistant context for system prompts.
        
        Returns:
            Context string with HA information
        """
        if self.homeassistant_content:
            return self.homeassistant_content
        
        # Fallback context if no prompt was loaded
        available_devices = self.get_available_devices()
        if available_devices:
            device_list = ", ".join(available_devices)
            return f"Home Assistant integration available. Controllable devices: {device_list}."
        
        return "Home Assistant integration available for device control."
    
    def get_home_keywords(self) -> List[str]:
        """
        Get list of keywords that indicate home automation requests.
        
        Returns:
            List of home automation keywords
        """
        return [
            "lights", "light", "lamp", "temperature", "temp", "heating", "cooling",
            "turn on", "turn off", "switch on", "switch off", "dim", "bright",
            "home", "house", "room", "bedroom", "kitchen", "living room",
            "thermostat", "climate", "hvac", "fan", "door", "window", "garage",
            "security", "alarm", "lock", "unlock", "sensor", "motion", "camera",
            "automation", "scene", "schedule"
        ] 