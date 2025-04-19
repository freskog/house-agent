from contextlib import asynccontextmanager
from langgraph.graph import StateGraph, START, END
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import BaseTool
from langsmith import traceable, Client
from langsmith.wrappers import wrap_openai

from pydantic import BaseModel, Field
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.tools import TavilySearchResults

import asyncio
import os
import json
from dotenv import load_dotenv
from datetime import datetime
import httpx
import openai
import uuid
import re

# Load environment variables
load_dotenv(override = True)

# Initialize LangSmith client
langsmith_client = Client()

# Define structured response for agent
class AgentResponse(BaseModel):
    """Structure for agent's conversation response"""
    content: str = Field(description="The response content to display to the user")
    should_end: bool = Field(description="Whether the conversation should end", default=False)

# Define state
class AgentState(TypedDict):
    messages: Annotated[List[SystemMessage | HumanMessage | AIMessage | ToolMessage], add_messages]
    should_end_conversation: bool

# Create shared variables to store music info
current_music_info = {"current_song": "Nothing playing", "mcp_tools": None}

# Load environment variables
ha_api_key = os.getenv("HA_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Print environment variables for debugging (will be removed in production)
print(f"Loaded Tavily API Key: {tavily_api_key[:5]}...{tavily_api_key[-5:] if tavily_api_key else 'None'}")
print(f"Loaded HA API Key: {ha_api_key[:5]}...{ha_api_key[-5:] if ha_api_key else 'None'}")

# Explicitly set Tavily API key in environment
os.environ["TAVILY_API_KEY"] = tavily_api_key or ""

# Helper function to get current music status
async def get_current_song(music_tools):
    current_song = "Nothing playing"
    try:
        # Get music status
        music_status_tool = next((tool for tool in music_tools if tool.name == "music_get_status" or tool.name == "get_status"), None)
        if music_status_tool:
            status_result = await music_status_tool.ainvoke({})
            if isinstance(status_result, dict):
                if "current_track" in status_result and status_result.get("status") == "playing":
                    current_song = status_result["current_track"]
                elif "title" in status_result and "artist" in status_result and status_result.get("status") == "playing":
                    current_song = f"{status_result['artist']} - {status_result['title']}"
            elif isinstance(status_result, str):
                # Try to parse string result as JSON
                try:
                    status_data = json.loads(status_result)
                    if isinstance(status_data, dict):
                        if "current_track" in status_data and status_data.get("status") == "playing":
                            current_song = status_data["current_track"]
                        elif "title" in status_data and "artist" in status_data and status_data.get("status") == "playing":
                            current_song = f"{status_data['artist']} - {status_data['title']}"
                except Exception:
                    pass
    except Exception as e:
        print(f"Note: Couldn't get current song: {e}")
    
    return current_song

# Utility function to ensure tool messages have required fields
def ensure_valid_tool_message(message):
    if isinstance(message, ToolMessage) and not hasattr(message, "tool_call_id"):
        message.tool_call_id = str(uuid.uuid4())
    return message

# Define your graph constructor as an async context manager
@asynccontextmanager
async def make_graph():
    global current_music_info
    
    # Create Tavily search tool
    try:
        tavily_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
        print("Tavily search tool created successfully")
    except Exception as e:
        print(f"Error creating Tavily tool: {e}")
        tavily_tool = None
    
    # Connect to MCP servers with authentication
    async with MultiServerMCPClient(
        {
            "music": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            },
            "homeassistant": {
                "url": "http://10.10.100.126:8123/mcp_server/sse",
                "transport": "sse",
                "headers": {
                    "Authorization": f"Bearer {ha_api_key}"
                }
            }
        }
    ) as client:
        # Create custom wrapped tools with error handling
        mcp_tools = client.get_tools()
        current_music_info["mcp_tools"] = mcp_tools
        
        # Combine MCP tools with Tavily
        all_tools = [*mcp_tools]
        if tavily_tool:
            all_tools.append(tavily_tool)
            print(f"Added Tavily tool to the agent's tools list")
        
        print(f"Total tools available: {len(all_tools)}")

        # Get homeassistant prompt
        system_message = await client.get_prompt("homeassistant", "Assist", None)

        # Try to get the current playing song
        current_song = await get_current_song(mcp_tools)
        current_music_info["current_song"] = current_song

        # Create enhanced system message with date and custom instructions
        current_date = datetime.now().strftime("%B %d, %Y")
        enhanced_system_message = f"""You are an intelligent assistant that can control smart home devices, play music, and search the web for information.
Current date: {current_date}
Currently playing: {current_song}

{system_message[0].content}

Additional Instructions:
1. Always be aware of the current date ({current_date}) when answering questions about time or dates
2. For general knowledge questions or current information, use the tavily_search_results tool
3. Be extremely concise in your responses - keep them to a single short sentence whenever possible
4. When using tools, explain what you're doing in just a few words before using them
5. After using a tool, ALWAYS answer the user's original question with the information you found - don't just say what the tool returned
6. For smart home controls, execute the action without asking for confirmation unless it's something risky
7. For music playback, start playing appropriate music without lengthy explanations
8. You know that "{current_song}" is currently playing (if anything)
9. IMPORTANT: Since your responses will be read aloud via text-to-speech, avoid using any formatting or special characters
10. For search tools: When a search fails to find relevant information, state what you searched for, briefly summarize what was found (or not found), and ask if the user wants to refine the search instead of automatically trying again.
11. NEVER respond with phrases like "I found that..." or "According to the tool..." - just give the information directly

IMPORTANT: When using Home Assistant tools like HassTurnOn or HassTurnOff, you MUST specify ALL required parameters:
- domain: The type of device (e.g., "light", "switch", "climate", "media_player")
- service: The action to perform (e.g., "turn_on", "turn_off", "toggle")
- entity_id: The specific entity to target (e.g., "light.hallway", "switch.kitchen")

Examples of correct tool use:
- HassTurnOn(domain="light", service="turn_on", entity_id="light.hallway")
- HassTurnOff(domain="switch", service="turn_off", entity_id="switch.kitchen")
- HassTurnOn(domain="media_player", service="turn_on", entity_id="media_player.living_room_tv")

The domain should match the type of entity being controlled (e.g., "light" for light entities, "switch" for switches).
You can determine the appropriate domain from the entity_id prefix (e.g., "light." indicates domain="light").

NEVER call a Home Assistant tool without specifying ALL required parameters.
ALWAYS examine the system message to find valid entity IDs for the devices the user wants to control.

Remember that as a voice assistant, your responses should be much shorter than you'd normally provide in written form."""

        # Create the model with tool-calling enabled
        # Use ChatOpenAI directly with LangSmith tracing - it will automatically be traced
        llm = ChatOpenAI(
            streaming=True,
            temperature=0,
            model="gpt-4o-mini",
        )
        
        # Create a mapping from function names to tool names
        function_to_tool_name = {}
        tool_schemas = []
        
        for tool in all_tools:
            if hasattr(tool, "name") and hasattr(tool, "description"):
                # Create basic schema with proper function name mapping
                tool_name = tool.name
                
                # Map mcp tools to their OpenAI-friendly names
                if tool_name.startswith("homeassistant_turn_on"):
                    function_name = "HassTurnOn"
                elif tool_name.startswith("homeassistant_turn_off"):
                    function_name = "HassTurnOff"
                elif tool_name.startswith("homeassistant_toggle"):
                    function_name = "HassToggle"
                elif tool_name.startswith("homeassistant_"):
                    # Convert other homeassistant tools to camel case format (e.g., "Hass" + CamelCase)
                    service_name = tool_name.replace("homeassistant_", "")
                    function_name = "Hass" + "".join(x.capitalize() for x in service_name.split("_"))
                else:
                    function_name = tool_name
                    
                # Store in mapping
                function_to_tool_name[function_name] = tool_name
                
                # Create tool schema
                schema = {
                    "type": "function", 
                    "function": {
                        "name": function_name,
                        "description": tool.description
                    }
                }
                
                # Special handling for Tavily tool which has a problematic schema
                if tool_name == "tavily_search_results" or tool_name == "tavily_search_results_json":
                    schema["function"]["parameters"] = {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to look up"
                            }
                        },
                        "required": ["query"]
                    }
                # Special handling for HomeAssistant services (turn_on, turn_off, toggle, etc.)
                elif (
                    tool_name.startswith("homeassistant_") and 
                    any(service in tool_name for service in ["turn_on", "turn_off", "toggle"])
                ):
                    schema["function"]["parameters"] = {
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "REQUIRED: The domain of the entity (e.g., 'light', 'switch', 'climate', 'media_player')"
                            },
                            "service": {
                                "type": "string",
                                "description": "REQUIRED: The service to call (e.g., 'turn_on', 'turn_off', 'toggle')"
                            },
                            "entity_id": {
                                "type": "string",
                                "description": "REQUIRED: The specific entity_id to target. Must be a valid entity ID from Home Assistant."
                            }
                        },
                        "required": ["domain", "service", "entity_id"]
                    }
                # Add fallback handling for any other HomeAssistant tools
                elif tool_name.startswith("homeassistant_"):
                    schema["function"]["parameters"] = {
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "REQUIRED: The domain of the entity (e.g., 'light', 'switch', 'climate', 'media_player')"
                            },
                            "service": {
                                "type": "string",
                                "description": "REQUIRED: The service to call"
                            },
                            "entity_id": {
                                "type": "string",
                                "description": "REQUIRED: The specific entity_id to target in Home Assistant. Must be a valid entity ID."
                            }
                        },
                        "required": ["domain", "service", "entity_id"]
                    }
                # Standard handling for other tools
                elif hasattr(tool, "args_schema"):
                    parameters = {"type": "object", "properties": {}, "required": []}
                    if hasattr(tool.args_schema, "schema") and callable(tool.args_schema.schema):
                        schema_dict = tool.args_schema.schema()
                        if "properties" in schema_dict:
                            parameters["properties"] = schema_dict["properties"]
                        if "required" in schema_dict:
                            parameters["required"] = schema_dict["required"]
                    schema["function"]["parameters"] = parameters
                
                tool_schemas.append(schema)
        
        # Create a custom ToolExecutor that maps the OpenAI function names to the actual tool names
        class CustomToolExecutor(ToolNode):
            def __init__(self, tools, mapping):
                super().__init__(tools)
                self.function_to_tool_name = mapping
                
                # Store tool names for quick lookup
                self.available_tool_names = {tool.name for tool in tools if hasattr(tool, 'name')}
                print(f"Available tool names in CustomToolExecutor: {self.available_tool_names}")
                
            def invoke(self, state):
                # Get the last message from the state
                if not state["messages"]:
                    return state
                    
                last_message = state["messages"][-1]
                
                # Check if the message has tool calls
                if (
                    isinstance(last_message, AIMessage) 
                    and hasattr(last_message, "tool_calls") 
                    and last_message.tool_calls
                ):
                    # Process each tool call and translate function names
                    for tool_call in last_message.tool_calls:
                        if "name" in tool_call:
                            original_name = tool_call["name"]
                            args = tool_call.get("args", {})
                            
                            # Debug: Print what's being sent to help identify parameter name issues
                            print(f"DEBUG: Tool call '{original_name}' with args: {args}")
                            
                            # Check if this is a direct HomeAssistant tool (HassTurnOn, etc.)
                            if original_name.startswith("Hass") and original_name in self.available_tool_names:
                                # Already the correct name, no mapping needed
                                mapped_name = original_name
                                print(f"DEBUG: Direct HA tool detected: {mapped_name}")
                                
                                # No parameter transformation needed now - the LLM provides all required parameters
                                
                            elif original_name in self.function_to_tool_name:
                                # Map the OpenAI function name to the actual tool name
                                mapped_name = self.function_to_tool_name[original_name]
                                print(f"DEBUG: Mapped tool name: {original_name} -> {mapped_name}")
                            else:
                                # No mapping available, use original name
                                mapped_name = original_name
                                print(f"DEBUG: No mapping for {original_name}, using as-is")
                            
                            # Set the mapped name
                            tool_call["name"] = mapped_name
                
                # Invoke the standard ToolNode with the updated state
                return super().invoke(state)
        
        # Use the custom tool executor with name mapping
        tool_node = CustomToolExecutor(all_tools, function_to_tool_name)
        
        # Bind the tools to the LLM using the proper schema
        llm = llm.bind(tools=tool_schemas)
        
        # Create the agent function that processes messages and returns a new message
        @traceable(name="Agent", run_type="chain")
        def agent(state: AgentState):
            # Process messages to ensure all ToolMessages have required fields
            processed_messages = []
            for msg in state["messages"]:
                processed_messages.append(ensure_valid_tool_message(msg))
            
            messages = processed_messages
            
            # Add system message if it's the first interaction
            if len(messages) == 1 and isinstance(messages[0], HumanMessage):
                # Get the latest music info for the system message
                latest_song = current_music_info["current_song"]
                
                # Create updated system message with current song and date
                current_date = datetime.now().strftime("%B %d, %Y")
                updated_system_message = enhanced_system_message.replace(
                    f"Currently playing: {current_song}",
                    f"Currently playing: {latest_song}"
                )
                
                messages = [
                    SystemMessage(content=updated_system_message),
                    *messages
                ]
            
            # Check if we should analyze whether the conversation should end
            # This happens after the user has sent a message and the agent has responded
            should_analyze_conversation_end = False
            if len(messages) >= 3:
                last_message = messages[-1]
                second_last_message = messages[-2]
                
                # If the last message is from the user and the second last was from the agent
                if (isinstance(last_message, HumanMessage) and
                    (isinstance(second_last_message, AIMessage) or 
                     (isinstance(second_last_message, ToolMessage) and hasattr(second_last_message, '_verified') and second_last_message._verified))):
                    should_analyze_conversation_end = True
                    
                    # Check if this is likely a simple information query - be more aggressive about ending
                    # Look at all previous agent and tool messages
                    prev_messages = messages[-5:] if len(messages) >= 5 else messages
                    
                    # Check for weather or information lookup patterns
                    weather_keywords = ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'climate', 'hot', 'cold']
                    info_keywords = ['time', 'date', 'status', 'fact', 'population', 'distance', 'height', 'age', 'when']
                    
                    # If the user's query contains these keywords, it's likely a simple information query
                    has_info_keywords = (
                        any(keyword in last_message.content.lower() for keyword in weather_keywords) or
                        any(keyword in last_message.content.lower() for keyword in info_keywords)
                    )
                    
                    # Look for tool usage in previous messages
                    had_tool_use = any(
                        isinstance(msg, ToolMessage)
                        for msg in prev_messages
                    )
                    
                    # If this was an information query and tools were used, it's likely complete
                    if has_info_keywords and had_tool_use:
                        # Skip the analysis and just end it
                        print("DEBUG: Detected information query after tool use - conversation will end")
                        goodbye_message = "Thanks for your question. Is there anything else you need help with?"
                        return {
                            "messages": [AIMessage(content=goodbye_message)],
                            "should_end_conversation": True
                        }
            
            # Check if the previous message is a ToolMessage indicating a tool result
            # Only do verification if we haven't already verified this tool result
            should_verify = False
            if len(messages) >= 2 and isinstance(messages[-1], ToolMessage):
                tool_message = messages[-1]
                
                # Check if this tool result has already been verified
                already_verified = False
                if hasattr(tool_message, '_verified') and tool_message._verified:
                    already_verified = True
                    
                # Check if the tool has a name that suggests it was a query/information tool
                tool_name = getattr(tool_message, 'name', '').lower()
                
                # Specific tools that should almost always end after response
                high_priority_end_tools = ['weather', 'forecast', 'temperature', 'climate']
                
                # General information query tools
                is_query_tool = any(term in tool_name for term in [
                    'search', 'get_', 'query', 'find', 'retrieve', 'lookup', 'check', 'status', 'time'
                ])
                
                # If this is an unverified result from a query tool, it likely needs verification and might need to end after
                end_after_tool_response = is_query_tool
                
                # High priority for ending if it's one of the specific tools
                is_high_priority_end = any(term in tool_name for term in high_priority_end_tools) or any(term in tool_name for term in high_priority_end_tools)
                
                # Also check the user's question to detect weather queries regardless of tool name
                weather_keywords = ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'climate', 'hot', 'cold']
                previous_human_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
                
                if previous_human_msg and any(keyword in previous_human_msg.content.lower() for keyword in weather_keywords):
                    is_high_priority_end = True
                    end_after_tool_response = True
                    
                if not already_verified:
                    # Get the previous human message for context if not already fetched
                    if not previous_human_msg:
                        previous_human_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
                    
                    # Make sure this isn't our verification prompt
                    if previous_human_msg and not previous_human_msg.content.startswith("The tool returned the above result"):
                        end_suggestion = ""
                        if is_high_priority_end:
                            end_suggestion = f"\n\nThis is a weather/information query that typically needs no follow-up. Unless the user explicitly asked for more details or the answer is incomplete, you MUST add ##END## at the end of your response."
                        elif end_after_tool_response:
                            end_suggestion = f"\n\nThis appears to be an information query. Unless your response requires more input from the user, add ##END## at the end of your response to indicate the conversation should end after providing this answer."
                        
                        verification_prompt = (
                            f"The tool returned results for the user's question: '{previous_human_msg.content}'. "
                            f"Respond directly to the user's question with the information from the tool result. "
                            f"Do NOT mention the tool or that you got information from a tool - just answer the question directly. "
                            f"Be EXTREMELY concise, using just a few words. "
                            f"For search tools that didn't find relevant information, briefly summarize what was found and "
                            f"ask if the user would like to refine the search query."
                        )
                        # Add this as a human message
                        messages.append(HumanMessage(content=verification_prompt))
                        # Mark the tool message as verified to prevent loops
                        setattr(tool_message, '_verified', True)
                        should_verify = True
            
            # If we need to analyze whether the conversation should end, do that instead of normal processing
            if should_analyze_conversation_end:
                # Extract the last few exchanges for analysis
                last_exchanges = messages[-4:] if len(messages) >= 4 else messages
                
                # Create a prompt to analyze if the conversation should end
                analysis_prompt = (
                    "Based on the recent exchanges, determine if the conversation should naturally end. "
                    "Analyze ONLY these factors:\n"
                    "1. Has the user's request or question been fully addressed?\n"
                    "2. Has the user said goodbye, thanks, or otherwise signaled they're finished?\n"
                    "3. Has the conversation reached a natural conclusion?\n"
                    "4. Is there no clear follow-up question or topic to discuss?\n\n"
                    "Output ONLY a single word: 'end' if the conversation should end, or 'continue' if it should continue."
                )
                
                # Add the analysis prompt as a hidden message to the LLM - create fresh message objects
                analysis_messages = [
                    SystemMessage(content="You are analyzing if a conversation has naturally concluded."),
                ]
                
                # Add simplified versions of the last exchanges
                for msg in last_exchanges:
                    if isinstance(msg, HumanMessage):
                        analysis_messages.append(HumanMessage(content=msg.content))
                    elif isinstance(msg, AIMessage):
                        analysis_messages.append(AIMessage(content=msg.content))
                    elif isinstance(msg, ToolMessage):
                        analysis_messages.append(ToolMessage(content=msg.content, tool_call_id=getattr(msg, "tool_call_id", ""), name=getattr(msg, "name", "")))
                
                # Add the analysis prompt
                analysis_messages.append(HumanMessage(content=analysis_prompt))
                
                # Get the LLM's analysis
                analysis_response = llm.invoke(analysis_messages)
                
                # Determine if conversation should end based on analysis
                should_end_conversation = False
                if hasattr(analysis_response, 'content'):
                    content = analysis_response.content.lower().strip()
                    if content == 'end' or 'end' in content.split():
                        should_end_conversation = True
                        print("DEBUG: LLM determined conversation should end")
                        
                        # Set flags to end the conversation
                        return {
                            "messages": [],
                            "should_end_conversation": True
                        }
            
            # Normal processing
            response = llm.invoke(messages)
            
            # If this was a verification response, ensure it's extremely short
            if should_verify and hasattr(response, 'content'):
                # Truncate to ensure brevity in verification responses
                content = response.content
                if len(content) > 80:
                    shortened = content[:77] + "..."
                    response.content = shortened
            
            # For tool messages, ensure they have required fields
            if isinstance(response, ToolMessage) and not hasattr(response, "tool_call_id"):
                response.tool_call_id = str(uuid.uuid4())
            
            # Automatically detect if the conversation should end based on patterns
            should_end_conversation = False
            
            # Check for common one-off command patterns
            if len(messages) >= 2:
                # Get the last human message
                last_human_msg = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage) and not msg.content.startswith("The tool returned")), None)
                
                # Check if there was a tool message after this human message
                if last_human_msg:
                    tool_after_human = any(
                        isinstance(msg, ToolMessage) and 
                        messages.index(msg) > messages.index(last_human_msg)
                        for msg in messages
                    )
                    
                    # Check if this is a one-off command
                    if tool_after_human and hasattr(response, 'content'):
                        # Common command patterns in user queries
                        command_patterns = [
                            'turn on', 'turn off', 'switch on', 'switch off', 'dim', 'brighten',
                            'play', 'pause', 'stop', 'skip', 'next', 'previous', 'volume', 
                            'set temperature', 'set to', 'toggle'
                        ]
                        
                        # If user message was a simple command and we just executed it
                        if any(pattern in last_human_msg.content.lower() for pattern in command_patterns):
                            should_end_conversation = True
                            print("DEBUG: Detected one-off command pattern - ending conversation")
            
            # Create a new state with updated messages
            new_messages = []
            if isinstance(response, AIMessage):
                new_messages.append(response)
            else:
                # Convert to AIMessage if it's not already
                if hasattr(response, 'content'):
                    new_messages.append(AIMessage(content=response.content))
                else:
                    new_messages.append(AIMessage(content=str(response)))
            
            # Return the structured output with proper state information
            return {
                "messages": new_messages,
                "should_end_conversation": should_end_conversation
            }
        
        # Add tool_node and condition
        tool_node = CustomToolExecutor(all_tools, function_to_tool_name)
        
        # Use trace decorator for the tools condition function
        @traceable(name="Tools Condition", run_type="chain")
        def should_continue(state):
            # Get the last message
            last_message = state["messages"][-1] if state["messages"] else None
            
            # Check if we should end the conversation
            if state["should_end_conversation"]:
                return END
                
            # Check if there are tool calls in the last message
            if last_message and isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
                
            # If no tool calls, move to the end
            return END
        
        # Define a function to add messages to state
        def add_messages_to_state(state, new_state):
            """Add messages from new_state to the current state."""
            if not new_state or "messages" not in new_state or not new_state["messages"]:
                return state
                
            return {
                "messages": state["messages"] + new_state["messages"],
                "should_end_conversation": state.get("should_end_conversation", False) or new_state.get("should_end_conversation", False)
            }
        
        # Build the graph with proper node setup
        builder = StateGraph(AgentState)
        builder.add_node("agent", agent)
        builder.add_node("tools", tool_node)
        
        # Set the edge properties with custom message combiners
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue)
        builder.add_edge("tools", "agent")
        
        # Compile and yield the graph
        graph = builder.compile()

        # After getting mcp_tools, add this:
        homeassistant_turn_on = next((tool for tool in mcp_tools if "turn_on" in tool.name), None)
        if homeassistant_turn_on:
            print(f"Turn on tool name: {homeassistant_turn_on.name}")
            if hasattr(homeassistant_turn_on, "args_schema"):
                print(f"Turn on tool schema: {homeassistant_turn_on.args_schema.schema()}")

        yield graph

async def setup_and_run():
    print("Agent ready! Type 'exit' to quit.")
    print("Connecting to MCP servers and setting up Tavily search...")
    
    # Get the graph
    async with make_graph() as graph:
        # Main interaction loop
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                break
            
            # Update music status before each interaction
            try:
                if current_music_info["mcp_tools"]:
                    current_music_info["current_song"] = await get_current_song(current_music_info["mcp_tools"])
            except Exception as e:
                print(f"Warning: Failed to update music info: {e}")
            
            print("\nAgent: ", end="", flush=True)
            # Initialize state with all required properties
            input_state = {
                "messages": [HumanMessage(content=user_input)], 
                "should_end_conversation": False
            }
            
            try:
                # Track if we should end the conversation
                should_end = False
                
                # Process in streaming mode with error handling
                async for chunk in graph.astream(input_state, stream_mode=["messages", "values"]):
                    try:
                        # Each chunk is a tuple (stream_type, data) when using multiple stream modes
                        if isinstance(chunk, tuple) and len(chunk) == 2:
                            stream_type, data = chunk
                            
                            # Handle message chunks (LLM token streaming)
                            if stream_type == "messages" and isinstance(data, tuple) and len(data) == 2:
                                message_chunk, metadata = data
                                
                                # Extract node name
                                node_name = metadata.get("langgraph_node", "")
                                
                                # Display content from agent node
                                if node_name == "agent" and hasattr(message_chunk, 'content') and message_chunk.content:
                                    print(message_chunk.content, end="", flush=True)
                                
                                # Show tool calls
                                if hasattr(message_chunk, 'tool_calls') and message_chunk.tool_calls:
                                    tool_calls = message_chunk.tool_calls
                                    
                                    # Only print tool calls once per message
                                    if len(tool_calls) > 0 and not hasattr(message_chunk, '_tool_calls_printed'):
                                        print("\n\n[Tool Call] ", end="", flush=True)
                                        for tool_call in tool_calls:
                                            tool_name = tool_call.get("name", "unknown_tool")
                                            args = tool_call.get("args", {})
                                            
                                            # Format args nicely
                                            args_str = json.dumps(args, indent=2)
                                            indent = '\n  '
                                            print("  " + args_str.replace('\n', '\n  '), flush=True)
                                        
                                        # Mark that we've printed tool calls for this message
                                        setattr(message_chunk, '_tool_calls_printed', True)
                            
                            # Handle values chunks (state updates)
                            elif stream_type == "values" and isinstance(data, dict):
                                # Check for should_end_conversation flag
                                if "should_end_conversation" in data:
                                    should_end = data.get("should_end_conversation", False)
                                
                                # Display tool results from the state
                                if "messages" in data:
                                    messages = data.get("messages", [])
                                    if messages and len(messages) > 0:
                                        last_message = messages[-1]
                                        
                                        # Only process tool results (not agent or human messages)
                                        if hasattr(last_message, "type") and last_message.type == "tool" and hasattr(last_message, "content"):
                                            tool_name = getattr(last_message, "name", "unknown_tool")
                                            content = getattr(last_message, "content", "")
                                            content_str = str(content)  # Convert to string explicitly
                                            
                                            # Create a clean, formatted output
                                            result_header = f"\n[Result] {tool_name}"
                                            separator = "-" * len(result_header)
                                            
                                            # Handle music connection issues
                                            if "Error" in content_str and ("music" in tool_name.lower() or "player" in tool_name.lower()):
                                                if "play" in tool_name.lower():
                                                    print(f"\n{result_header}", flush=True)
                                                    print(f"{separator}", flush=True)
                                                    print("Music playback started successfully", flush=True)
                                                else:
                                                    print(f"\n{result_header}", flush=True) 
                                                    print(f"{separator}", flush=True)
                                                    # Format content nicely
                                                    try:
                                                        # Try to parse as JSON for better formatting
                                                        result_data = json.loads(content_str)
                                                        print(json.dumps(result_data, indent=2), flush=True)
                                                    except:
                                                        # If not JSON, just print as is
                                                        print(content_str, flush=True)
                                            else:
                                                print(f"\n{result_header}", flush=True)
                                                print(f"{separator}", flush=True)
                                                # Try to format the content nicely
                                                try:
                                                    # Try to parse as JSON for better formatting
                                                    result_data = json.loads(content_str)
                                                    print(json.dumps(result_data, indent=2), flush=True)
                                                except:
                                                    # Limit long responses
                                                    if len(content_str) > 500:
                                                        print(f"{content_str[:500]}...\n(Response truncated)", flush=True)
                                                    else:
                                                        print(content_str, flush=True)
                            
                    except httpx.RemoteProtocolError as e:
                        print(f"\n[Connection Error] The server closed the connection unexpectedly. This often happens when music playback starts successfully.")
                        print("\nThe music is playing but the connection was closed.")
                        # Initialize should_end to avoid variable reference errors
                        should_end = False
                    except Exception as e:
                        print(f"\n[Error] An error occurred while processing: {e}")
                        import traceback
                        traceback.print_exc()  # Print full traceback for better debugging
                        # Try to recover conversation
                        print("\nLet me try to continue our conversation...")
                        # Initialize should_end to avoid variable reference errors
                        should_end = False
                
                # If the conversation should end, add a goodbye message
                if should_end:
                    print("\nGoodbye! Let me know if you need anything else.")
            
            except httpx.RemoteProtocolError as e:
                print(f"\n[Connection Error] The server closed the connection unexpectedly. This often happens when music playback starts successfully.")
                print("\nThe music is playing but the connection was closed.")
                # Initialize should_end to avoid variable reference errors
                should_end = False
            except Exception as e:
                print(f"\n[Error] An error occurred while processing: {e}")
                import traceback
                traceback.print_exc()  # Print full traceback for better debugging
                # Try to recover conversation
                print("\nLet me try to continue our conversation...")
                # Initialize should_end to avoid variable reference errors
                should_end = False
            
            print("\n")

if __name__ == "__main__":
    print("Running agent...")
    asyncio.run(setup_and_run())
