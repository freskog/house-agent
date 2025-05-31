from contextlib import asynccontextmanager
from langgraph.graph import StateGraph, START, END
from typing import Annotated, List, Dict, Any, Optional, Generator
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import BaseTool
from langsmith import traceable, Client
from langsmith.wrappers import wrap_openai

from pydantic import BaseModel, Field
from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
from langchain_community.tools import TavilySearchResults

# Import Spotify integration
from spotify import SpotifyClient, create_spotify_tools
from spotify.tools import current_music_state

import asyncio
import os
import json
from dotenv import load_dotenv
from datetime import datetime
import openai
import uuid
import time
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("house_agent")

# Load environment variables
load_dotenv(override = True)

# Initialize LangSmith client
langsmith_client = Client()

# Set up in-memory caching for LangChain to avoid those warnings
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
# Configure in-memory cache
set_llm_cache(InMemoryCache())
print("Configured LangChain cache")

# Define structured output format for the agent
class AgentResponse(BaseModel):
    """Structured format for agent responses"""
    messages: List[AIMessage] = Field(description="Messages to add to the conversation")

# Define state
class AgentState(TypedDict):
    messages: Annotated[List[SystemMessage | HumanMessage | AIMessage | ToolMessage], add_messages]

# Configure OpenAI caching and endpoints
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# Configure the OpenAI client with persistent caching enabled
# This allows responses to be cached across requests
if openai_api_key:
    openai.api_key = openai_api_key
    
    # Configure the base URL if specified
    if openai_base_url:
        openai.base_url = openai_base_url
        print(f"Using custom OpenAI base URL: {openai_base_url}")
        
    # Configure default timeout
    openai.timeout = 30.0
    
    print("OpenAI client configured with API key and caching support")
else:
    print("WARNING: No OpenAI API key found in environment variables")

# Global variable to track the system message for reuse
cached_system_message = None
# Global dictionary to track current music info - accessible across all invocations
current_music_info = {
    "current_song": "Nothing playing",
    "mcp_tools": []
}

# Load environment variables
ha_api_key = os.getenv("HA_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Print environment variables for debugging (will be removed in production)
print(f"Loaded Tavily API Key: {tavily_api_key[:5]}...{tavily_api_key[-5:] if tavily_api_key else 'None'}")
print(f"Loaded HA API Key: {ha_api_key[:5]}...{ha_api_key[-5:] if ha_api_key else 'None'}")

# Explicitly set Tavily API key in environment
os.environ["TAVILY_API_KEY"] = tavily_api_key or ""

# Helper function to pretty print objects
def pretty_print(obj: Any, indent: int = 2) -> None:
    """Pretty print an object as JSON or fallback to string representation."""
    try:
        if isinstance(obj, (dict, list)):
            print(json.dumps(obj, indent=indent))
        elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            print(json.dumps(obj.to_dict(), indent=indent))
        elif hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            print(json.dumps(obj.model_dump(), indent=indent))
        elif hasattr(obj, "__dict__"):
            print(json.dumps({k: v for k, v in obj.__dict__.items() if not k.startswith("_")}, indent=indent))
        else:
            print(str(obj))
    except Exception:
        print(f"<Non-serializable object of type {type(obj).__name__}>")

# Utility function to ensure tool messages have required fields
def ensure_valid_tool_message(message):
    if isinstance(message, ToolMessage) and not hasattr(message, "tool_call_id"):
        message.tool_call_id = str(uuid.uuid4())
    return message

# Define your graph constructor as an async context manager
@asynccontextmanager
async def make_graph():
    global current_music_info
    global cached_system_message
    
    print("Starting agent initialization...")
    start_time = time.time()
    
    try:
        # Set up MCP connections similar to ha_langchain_mcp.py
        connections = {}
        
        # Only add Home Assistant if we have an API key
        if ha_api_key:
            # Home Assistant MCP server URL
            ha_url = "http://10.10.100.126:8123/mcp_server/sse"
            
            # Set up SSE connection configuration for Home Assistant
            ha_connection: SSEConnection = {
                "transport": "sse",
                "url": ha_url,
                "headers": {"Authorization": f"Bearer {ha_api_key}"},
                "timeout": 10,             # HTTP connection timeout
                "sse_read_timeout": 60,    # SSE read timeout
                "session_kwargs": {}       # Additional session parameters if needed
            }
            
            # Add Home Assistant connection
            connections["home_assistant"] = ha_connection
            print("Added Home Assistant connection to MCP client")
        else:
            print("No HA_API_KEY found - Home Assistant connection will not be available")
        
        # Initialize client with connections
        print(f"Initializing MCP client with {len(connections)} connections")
        client = MultiServerMCPClient(connections=connections)
        
        # Create Tavily search tool if API key is available
        tavily_tool = None
        if tavily_api_key:
            # Use Tavily search tool
            tavily_tool = TavilySearchResults(max_results=3)
            print("Tavily search tool created")
        else:
            print("Warning: No Tavily API key found - web search tool will not be available")
        
        # Create Spotify client and tools
        spotify_tools = []
        try:
            spotify_client = SpotifyClient()
            spotify_tools = create_spotify_tools(spotify_client)
            print(f"Created {len(spotify_tools)} Spotify tools")
        except Exception as e:
            print(f"Warning: Could not initialize Spotify tools: {e}")
            print("Spotify functionality will not be available")
        
        # Get tools from MCP client - these are already properly formatted LangChain tools
        try:
            print("Getting tools from MCP client...")
            mcp_tools = await client.get_tools()
            print(f"Loaded {len(mcp_tools)} tools from MCP client")
            print("Tool types:")
            for i, tool in enumerate(mcp_tools[:5]):  # Just log first 5 to avoid spam
                print(f"{i+1}. {tool.name}: {type(tool).__name__}")
        except Exception as e:
            print(f"Error loading MCP tools: {e}")
            mcp_tools = []
            
        current_music_info["mcp_tools"] = mcp_tools
        
        # Combine all tools: MCP tools + Spotify tools + Tavily
        all_tools = [*mcp_tools, *spotify_tools]
        if tavily_tool:
            all_tools.append(tavily_tool)
            print(f"Added Tavily tool to the agent's tools list")
        
        print(f"Total tools available: {len(all_tools)}")
        if spotify_tools:
            spotify_tool_names = [tool.name for tool in spotify_tools]
            print(f"Spotify tools: {spotify_tool_names}")

        # Try to load prompt from MCP server (simplifying the approach from previous version)
        homeassistant_content = None
        try:
            # Get prompts from Home Assistant
            print("Listing available prompts from Home Assistant...")
            async with client.session("home_assistant") as session:
                prompts = await session.list_prompts()
                
                # Handle different return types
                prompts_list = prompts.prompts if hasattr(prompts, "prompts") else prompts
                
                if prompts_list:
                    print(f"Found {len(prompts_list)} prompts:")
                    for i, prompt in enumerate(prompts_list):
                        prompt_name = prompt.name if hasattr(prompt, "name") else prompt.get("name", "Unknown")
                        print(f"{i+1}. {prompt_name}")
                        
                    # Try to get the first prompt
                    first_prompt = prompts_list[0]
                    prompt_name = first_prompt.name if hasattr(first_prompt, "name") else first_prompt.get("name", "Unknown")
                    print(f"Getting content for prompt '{prompt_name}'...")
                    try:
                        prompt_details = await session.get_prompt(prompt_name)
                        if hasattr(prompt_details, "content"):
                            homeassistant_content = prompt_details.content
                        elif isinstance(prompt_details, dict):
                            homeassistant_content = prompt_details.get("content")
                        if homeassistant_content:
                            print(f"Successfully loaded prompt content ({len(homeassistant_content)} chars)")
                        else:
                            print("No content found in prompt details")
                    except Exception as e:
                        print(f"Failed to get prompt details: {e}")
                else:
                    print("No prompts found")
        except Exception as e:
            print(f"Failed to list prompts: {e}")
        
        # Initialize current song as placeholder (will be replaced by Spotify integration)
        current_song = "Nothing playing"
        current_music_info["current_song"] = current_song

        # Create enhanced system message with date and custom instructions
        current_date = datetime.now().strftime("%B %d, %Y")
        enhanced_system_message = f"""You are a helpful AI assistant for a smart home. Today is {current_date}.

Currently playing: {current_song}

{homeassistant_content if homeassistant_content else ''}

You have access to various tools to help control the home and get information. Use these tools when appropriate to help the user.

Music Control (Spotify Web API):
- Use get_current_song to check what's currently playing
- Use play_music to search and play tracks, albums, or artists
- Use pause_music, stop_music, next_track, previous_track for playback control
- Use set_volume to adjust volume (0-100)
- Use search_music to find specific tracks and get their URIs
- Use get_spotify_devices to see available playback devices

When the user asks about music:
1. Use the appropriate Spotify tools for control
2. For "what's playing" queries, use get_current_song and return just the track info
3. For play requests, search if needed, then play the track
4. For stop/pause requests, use stop_music or pause_music
5. Provide helpful feedback about what actions were taken

For all other requests:
1. Be helpful and concise
2. Use tools when needed
3. Don't make assumptions about what the user wants
4. Ask for clarification if needed"""

        # Cache the system message
        cached_system_message = enhanced_system_message
        print(f"System message created and cached ({len(enhanced_system_message)} chars)")

        # Create the model with tool-calling enabled
        # Use ChatOpenAI directly with LangSmith tracing - it will automatically be traced
        llm = ChatOpenAI(
            streaming=True,
            temperature=0,
            model="gpt-4o-mini",
            cache=True  # Enable response caching
        )
        
        # Log available tools for debugging
        tool_names = {tool.name for tool in all_tools if hasattr(tool, 'name')}
        print(f"Available tool names: {tool_names}")
                
        # Bind the tools to the LLM directly - no need for custom schema conversion
        llm = llm.bind_tools(all_tools)
        
        # Use standard ToolNode without any mapping - the MCP server provides tools with the correct naming already
        tool_node = ToolNode(all_tools)
        
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
                # Get the latest music info from the Spotify tools state
                latest_song = current_music_state.get("current_song", "Nothing playing")
                
                # Create updated system message with current song and date
                current_date = datetime.now().strftime("%B %d, %Y")
                updated_system_message = cached_system_message.replace(
                    f"Currently playing: {current_song}",
                    f"Currently playing: {latest_song}"
                )
                
                messages = [
                    SystemMessage(content=updated_system_message),
                    *messages
                ]

            # Always create a new thread_id for now
            # This will be overridden by the config if provided when the graph is invoked
            thread_id = str(uuid.uuid4())
            
            # Normal processing with thread_id
            # This will be passed through to LangSmith for tracing
            response = llm.invoke(messages, config={"thread_id": thread_id})
            
            # For tool messages, ensure they have required fields
            if isinstance(response, ToolMessage) and not hasattr(response, "tool_call_id"):
                response.tool_call_id = str(uuid.uuid4())
            
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
            
            # Return structured response
            return AgentResponse(
                messages=new_messages
            ).model_dump()
        
        # Use trace decorator for the tools condition function
        @traceable(name="Tools Condition", run_type="chain")
        def should_continue(state):
            # Get the last message
            last_message = state["messages"][-1] if state["messages"] else None
            
            # Check if there are tool calls in the last message
            if last_message and isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
                
            # If no tool calls, move to the end
            return "end"
        
        # Add conditional routing after tool execution
        @traceable(name="Post Tools Routing", run_type="chain")
        def should_continue_after_tools(state):
            """Determine if we should continue to agent or end directly after tool execution."""
            # Get the last few messages to find the tool result
            messages = state["messages"]
            if not messages:
                return "agent"
            
            # Look for the most recent ToolMessage
            for message in reversed(messages):
                if isinstance(message, ToolMessage):
                    # Check if it was a music control command by looking at the name field first
                    tool_name = getattr(message, 'name', '')
                    
                    # If name is not available, we need to find the tool name another way
                    # We'll look at the previous AIMessage to see what tool was called
                    if not tool_name:
                        # Find the AIMessage that called this tool
                        message_index = messages.index(message)
                        for i in range(message_index - 1, -1, -1):
                            prev_msg = messages[i]
                            if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                                # Match by tool_call_id
                                for tool_call in prev_msg.tool_calls:
                                    if hasattr(tool_call, 'id') and tool_call.id == message.tool_call_id:
                                        tool_name = getattr(tool_call, 'name', '')
                                        break
                                if tool_name:
                                    break
                    
                    # Check if it was a music control command
                    if tool_name in ["play_music", "pause_music", "stop_music", "next_track", "previous_track", "set_volume"]:
                        # Check if the command succeeded (empty content or no error message)
                        content = getattr(message, 'content', '')
                        if not content or not any(error_word in content.lower() for error_word in ["failed", "error", "no tracks found"]):
                            print(f"Music command '{tool_name}' succeeded silently, routing to END")
                            return "end"
                        else:
                            print(f"Music command '{tool_name}' failed: {content}, routing to agent")
                    break
            
            # Default: continue to agent for all other cases
            return "agent"
        
        # Define a function to add messages to state
        def add_messages_to_state(state, new_state):
            """Add messages from new_state to the current state."""
            if not new_state or "messages" not in new_state or not new_state["messages"]:
                return state
            
            return {
                "messages": state["messages"] + new_state["messages"],
            }
        
        # Build the graph with proper node setup
        builder = StateGraph(AgentState)
        
        # Add nodes
        builder.add_node("agent", agent)
        builder.add_node("tools", tool_node)
        
        # Set the edge properties
        builder.add_edge(START, "agent")
        
        # Add conditional edges
        builder.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # Add conditional routing from tools - either back to agent or directly to end
        builder.add_conditional_edges(
            "tools",
            should_continue_after_tools,
            {
                "agent": "agent",
                "end": END
            }
        )
        
        # Compile and yield the graph
        graph = builder.compile(
            # Set a higher recursion limit to handle complex workflows
            # but still prevent infinite loops
            debug=True
        )

        # Check if we have a Home Assistant turn_on tool and print its details
        turn_on_tool = next((tool for tool in all_tools if "turn_on" in getattr(tool, "name", "")), None)
        if turn_on_tool:
            print(f"Turn on tool name: {turn_on_tool.name}")
            if hasattr(turn_on_tool, "args_schema"):
                # Try to print schema details safely
                try:
                    schema_info = {}
                    if hasattr(turn_on_tool.args_schema, "schema"):
                        schema_info = turn_on_tool.args_schema.schema()
                    elif isinstance(turn_on_tool.args_schema, dict):
                        schema_info = turn_on_tool.args_schema
                    else:
                        schema_info = {"type": str(type(turn_on_tool.args_schema))}
                    print(f"Turn on tool schema: {json.dumps(schema_info, indent=2)}")
                except Exception as e:
                    print(f"Could not print schema details: {e}")
                    print(f"Schema type: {type(turn_on_tool.args_schema).__name__}")

        # Perform a warm-up call to eagerly load the system message into cache
        print("Performing warm-up call to load system prompt into cache...")
        warmup_start_time = time.time()
        try:
            # Create a simple test message
            warmup_messages = [
                SystemMessage(content=cached_system_message),
                HumanMessage(content="hello")
            ]
            
            # Make a non-streaming call to ensure it's fully loaded and cached
            # Create a non-streaming version for the warmup
            warmup_llm = ChatOpenAI(
                streaming=False,
                temperature=0,
                model="gpt-4o-mini",
                cache=True
            ).bind_tools(all_tools)
            
            # Issue a warmup call
            _ = warmup_llm.invoke(warmup_messages)
            warmup_elapsed = time.time() - warmup_start_time
            print(f"Warm-up call completed successfully in {warmup_elapsed:.2f} seconds - system prompt loaded into cache")
        except Exception as e:
            print(f"Warm-up call failed, but we'll continue anyway: {e}")
            
        total_init_time = time.time() - start_time
        print(f"Agent graph initialization completed in {total_init_time:.2f} seconds")

        # Yield the graph
        yield graph
    except Exception as e:
        print(f"Error during agent initialization: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise the exception after logging
        raise

if __name__ == "__main__":
    print("Running agent...")
    # Use async main function to properly use the context manager
    async def run_agent():
        async with make_graph() as graph:
            print("Agent graph created")
            # Keep the process running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("Agent stopped")
    
    asyncio.run(run_agent())
