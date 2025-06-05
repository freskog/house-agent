"""
Music node for handling music and audio control requests.

This node manages all music-related functionality including:
- Music playback control (play, pause, stop, next, previous)
- Volume control
- Music search and recommendations
- Radio station creation
- Current song status
- Silent operation for successful music commands
- LLM-powered function calling for tool selection

Currently integrates with Spotify but designed to support multiple music services.
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from .base_node import BaseNode, AgentState

from spotify import create_spotify_tools
from spotify.tools import current_music_state, update_music_state
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class MusicNode(BaseNode):
    """Node for handling music and audio control requests with LLM function calling."""
    
    # Define music tool names for routing decisions
    MUSIC_TOOLS = [
        "play_music", "pause_music", "stop_music", "next_track", 
        "previous_track", "set_volume", "create_radio_station",
        "get_current_song", "search_music", "get_spotify_devices"
    ]
    
    # Music control commands that should end silently on success
    SILENT_COMMANDS = [
        "play_music", "pause_music", "stop_music", "next_track", 
        "previous_track", "set_volume", "create_radio_station"
    ]
    
    def __init__(self):
        """Initialize Music node with music service tools and LLM with function calling."""
        try:
            music_tools = create_spotify_tools()
            super().__init__(music_tools, "Music")
            
            # Initialize LLM with tools bound for function calling
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                streaming=False
            ).bind_tools(music_tools)
            
            self.logger.info(f"Initialized Music node with {len(music_tools)} tools and function calling")
        except Exception as e:
            self.logger.error(f"Failed to initialize music tools: {e}")
            # Initialize with empty tools if music setup fails
            super().__init__([], "Music")
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                streaming=False
            )
            self.logger.warning("Music node initialized without tools due to setup failure")
    
    def should_handle_request(self, message: str) -> bool:
        """
        Determine if this node should handle the music-related request.
        
        Args:
            message: User message to evaluate
            
        Returns:
            True if this node should handle the request, False otherwise
        """
        message_lower = message.lower()
        
        # Music-related keywords that indicate music handling
        music_keywords = [
            "play", "music", "song", "volume", "spotify", "pause", "stop",
            "next", "previous", "skip", "radio", "album", "artist", "track",
            "sound", "audio", "listen", "hear", "loud", "quiet", "louder",
            "softer", "turn up", "turn down", "what's playing", "now playing"
        ]
        
        return any(keyword in message_lower for keyword in music_keywords)
    
    async def handle_request(self, state: AgentState, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle music-related requests using LLM function calling.
        
        Args:
            state: Current agent state
            config: Optional configuration
            
        Returns:
            Updated state after handling the request
        """
        if not self.tools:
            self.logger.warning("No music tools available")
            return self.create_response([
                AIMessage(content="Sorry, music functionality is not available right now.")
            ])
        
        # Get context about who called us
        called_by = state.get("called_by", "router")
        user_message = self.get_last_user_message(state)
        
        if not user_message:
            return self.create_response([
                AIMessage(content="I didn't receive a clear music request.")
            ])
        
        # Check if this is a multi-domain request that should be escalated
        if called_by == "router" and self._should_escalate(user_message.content):
            return await self._handle_escalation(user_message.content, state)
        
        # Use LLM with function calling to handle the request
        return await self._handle_with_function_calling(user_message.content, state)
    
    def _should_escalate(self, user_request: str) -> bool:
        """Check if the request involves multiple domains and should be escalated."""
        request_lower = user_request.lower()
        
        # Check for music keywords
        has_music = any(keyword in request_lower for keyword in [
            "play", "music", "song", "volume", "spotify", "pause", "stop",
            "next", "previous", "skip", "radio", "album", "artist", "track"
        ])
        
        # Check for non-music keywords
        has_non_music = any(keyword in request_lower for keyword in [
            "light", "lights", "temperature", "heat", "cool", "search", "find",
            "weather", "news", "search for", "look up", "turn on", "turn off"
        ])
        
        return has_music and has_non_music
    

    
    async def _handle_with_function_calling(self, user_request: str, state: AgentState) -> Dict[str, Any]:
        """Handle music request using LLM function calling."""
        current_date = datetime.now().strftime("%B %d, %Y")
        current_song = current_music_state.get("current_song", "Nothing playing")
        
        # Create system prompt for music handling
        system_prompt = f"""You are a music specialist. Today is {current_date}.
Currently playing: {current_song}

You have access to music control tools. For user requests:

1. For playing music, ALWAYS use the play_music tool with proper context:
   - Swedish content: {{"context": {{"type": "artist|track|album|genre", "market": "SE"}}}}
   - Other content: {{"context": {{"type": "artist|track|album|genre", "market": "IE"}}}}

2. For control commands (pause, stop, next, etc.), use the appropriate tools

3. For information requests, use get_current_song or search_music

4. Always use the available tools - don't just describe what to do

Handle the user's music request by calling the appropriate tools."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_request)
        ]
        
        try:
            # Let the LLM generate function calls
            response = await self.llm.ainvoke(messages)
            
            # Check if the LLM made tool calls
            if response.tool_calls:
                self.logger.info(f"LLM generated {len(response.tool_calls)} tool calls")
                
                # Execute the tool calls
                tool_results = await self._execute_tool_calls(response.tool_calls, state)
                
                # Process results and determine response
                response_text = self._process_tool_results(tool_results, response.tool_calls)
                return self.create_response([AIMessage(content=response_text)])
            else:
                # LLM didn't call tools, return its response
                if response.content:
                    return self.create_response([AIMessage(content=response.content)])
                else:
                    return self.create_response([AIMessage(content="I wasn't sure how to handle that music request.")])
                    
        except Exception as e:
            self.logger.error(f"Error in function calling: {e}")
            return self.create_response([
                AIMessage(content="I encountered an error processing your music request.")
            ])
    
    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]], state: AgentState) -> List[Dict[str, Any]]:
        """Execute the LLM-generated tool calls."""
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            # Find the tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                results.append({
                    "tool_call": tool_call,
                    "error": f"Tool {tool_name} not found",
                    "content": f"Error: Tool {tool_name} is not available"
                })
                continue
            
            try:
                # Handle tools that expect no arguments
                if tool_name in ["pause_music", "stop_music", "next_track", "previous_track", "get_current_song", "get_spotify_devices"]:
                    # These tools expect a string, not a dict
                    tool_input = ""
                else:
                    # These tools expect arguments dict
                    tool_input = tool_args
                
                # Execute the tool
                result = await tool.ainvoke(tool_input) if hasattr(tool, 'ainvoke') else tool.invoke(tool_input)
                results.append({
                    "tool_call": tool_call,
                    "tool_name": tool_name,
                    "content": result
                })
            except Exception as e:
                self.logger.error(f"Error executing tool {tool_name}: {e}")
                results.append({
                    "tool_call": tool_call,
                    "error": str(e),
                    "content": f"Error executing {tool_name}: {str(e)}"
                })
        
        return results
    
    def _process_tool_results(self, tool_results: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]]) -> str:
        """Process tool results into a human-readable response."""
        if not tool_results:
            return "No results from music tools."
        
        # Check if all tools were silent commands that succeeded
        silent_commands = []
        responses = []
        errors = []
        
        for result in tool_results:
            tool_name = result.get("tool_name", "")
            content = result.get("content", "")
            error = result.get("error")
            
            if error:
                errors.append(f"Error: {error}")
            elif tool_name in self.SILENT_COMMANDS:
                # Silent command - check if it succeeded (empty content means success)
                if content == "":
                    silent_commands.append(tool_name)
                else:
                    responses.append(content)
            elif content:
                # Non-silent command with content
                responses.append(content)
        
        # If there were errors, return them
        if errors:
            return " ".join(errors)
        
        # If all were successful silent commands, return empty (silent response)
        if silent_commands and not responses:
            return ""
        
        # Return any non-silent responses
        if responses:
            return " ".join(responses)
        
        # Fallback
        return "Music command completed."
    
    # Legacy methods for backward compatibility
    def should_end_silently(self, state: AgentState) -> bool:
        """Legacy method - now handled by tool result processing."""
        return False
    
    def _extract_tool_name(self, tool_message: ToolMessage, all_messages: List) -> str:
        """Legacy method for extracting tool names."""
        return getattr(tool_message, 'name', '')
    
    @property
    def current_song(self) -> str:
        """Get the current song from the global music state."""
        return current_music_state.get("current_song", "Nothing playing")
    
    def update_music_state(self, new_state: str):
        """Update the global music state."""
        update_music_state(new_state)
        self.logger.debug(f"Updated music state: {new_state}")
    
    def create_silent_response(self) -> Dict[str, Any]:
        """Create a silent response for successful music commands."""
        return self.create_response([AIMessage(content="")])
    
    def get_music_keywords(self) -> List[str]:
        """Get list of keywords that indicate music-related requests."""
        return [
            "play", "music", "song", "volume", "spotify", "pause", "stop",
            "next", "previous", "skip", "radio", "album", "artist", "track",
            "sound", "audio", "listen", "hear", "loud", "quiet", "louder",
            "softer", "turn up", "turn down", "what's playing", "now playing",
            "bass", "treble", "equalizer", "shuffle", "repeat", "playlist"
        ] 