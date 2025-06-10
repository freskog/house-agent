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
from langchain_groq import ChatGroq
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
            self.llm = ChatGroq(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0,
                streaming=True
            ).bind_tools(music_tools)
            
            self.logger.info(f"Initialized Music node with {len(music_tools)} tools and function calling")
        except Exception as e:
            self.logger.error(f"Failed to initialize music tools: {e}")
            # Initialize with empty tools if music setup fails
            super().__init__([], "Music")
            self.llm = ChatGroq(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0,
                streaming=True
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
    
    async def _handle_escalation(self, user_request: str, state: AgentState) -> Dict[str, Any]:
        """Handle escalation to agent for multi-domain requests."""
        return self.create_escalation_response(
            reason="Multi-domain request detected",
            domains=["music"],
            original_request=user_request
        )

    
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
                response_text = await self._process_tool_results(tool_results, response.tool_calls, user_request)
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
    
    async def _process_tool_results(self, tool_results: List[Dict[str, Any]], tool_calls: List[Dict[str, Any]], user_request: str = "") -> str:
        """Process tool results into a human-readable response."""
        if not tool_results:
            return "No results from music tools."
        
        # Check if all tools were silent commands that succeeded
        silent_commands = []
        responses = []
        errors = []
        raw_informational_content = []
        
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
                # Check if this is informational content that could benefit from LLM summarization
                if tool_name in ["get_current_song", "search_music", "get_spotify_devices"] and len(str(content)) > 50:
                    raw_informational_content.append(str(content))
                else:
                    responses.append(content)
        
        # If there were errors, return them
        if errors:
            return " ".join(errors)
        
        # If all were successful silent commands, return empty (silent response)
        if silent_commands and not responses and not raw_informational_content:
            return ""
        
        # Use LLM to summarize informational content if available
        if raw_informational_content and self._is_informational_music_query(user_request):
            try:
                summary = await self._create_music_summary(raw_informational_content, user_request)
                if summary:
                    responses.append(summary)
            except Exception as e:
                self.logger.error(f"Error creating music summary: {e}")
                # Fallback to original content
                responses.extend(raw_informational_content)
        else:
            # Add raw content if not summarized
            responses.extend(raw_informational_content)
        
        # Return any responses
        if responses:
            return " ".join(responses)
        
        # Fallback
        return "Music command completed."
    
    def _is_informational_music_query(self, user_request: str) -> bool:
        """Check if the user request is asking for music information rather than performing an action."""
        if not user_request:
            return False
        
        query_lower = user_request.lower()
        
        # Check for question words and patterns for music info
        info_patterns = [
            "what's", "what is", "which", "who", "is", "are", "current", "currently",
            "playing", "now playing", "what song", "what music", "tell me", "show me",
            "status", "info", "information", "details", "search for", "find"
        ]
        
        # Action patterns that should remain silent
        action_patterns = [
            "play ", "pause", "stop", "next", "previous", "skip", "volume",
            "turn up", "turn down", "louder", "softer", "set volume"
        ]
        
        # If it contains action patterns, it's not informational
        if any(pattern in query_lower for pattern in action_patterns):
            return False
        
        # If it contains info patterns, it's informational
        return any(pattern in query_lower for pattern in info_patterns)
    
    async def _create_music_summary(self, raw_content: List[str], user_request: str) -> str:
        """Create a human-readable summary of music information using LLM."""
        if not raw_content:
            return ""
        
        combined_content = "\n\n".join(raw_content)
        
        system_prompt = f"""You are summarizing music/Spotify information for a voice assistant.

User asked: "{user_request}"

Music data:
{combined_content[:800]}

Create a concise, natural response that:
- Directly answers the user's question about music/what's playing
- Is easy to understand when spoken aloud
- Focuses on the most relevant information (song name, artist, album)
- Keeps it under 2-3 sentences
- Sounds conversational and natural

If the user asked about what's playing, focus on current song/artist.
If they searched for music, mention the results found.

Response:"""

        try:
            summary_llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0, streaming=True, max_tokens=100)
            messages = [SystemMessage(content=system_prompt)]
            response = await summary_llm.ainvoke(messages)
            
            if response.content:
                return response.content.strip()
            else:
                return ""
                
        except Exception as e:
            self.logger.error(f"Error in music LLM summarization: {e}")
            return ""
    
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