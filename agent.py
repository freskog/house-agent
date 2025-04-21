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

import asyncio
import os
import json
from dotenv import load_dotenv
from datetime import datetime
import httpx
import openai
import uuid
import re
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

# Helper function to get current music status
async def get_current_song(client_session, music_tools):
    current_song = "Nothing playing"
    try:
        # Get music status
        music_status_tool = next((tool for tool in music_tools if tool.name == "music_get_status" or tool.name == "get_status"), None)
        if music_status_tool:
            logger.debug(f"Using music status tool: {music_status_tool.name}")
            try:
                # Use arun() for StructuredTools
                status_result = await music_status_tool.arun({})
                logger.debug(f"Music status raw result: {status_result}")
            except Exception as e:
                logger.debug(f"arun() failed, trying direct tool call: {e}")
                # Try direct call through MCP session as fallback
                try:
                    # Use direct call through the provided client session
                    if client_session and hasattr(client_session, "sessions") and "music" in client_session.sessions:
                        status_result = await client_session.sessions["music"].call_tool(music_status_tool.name, {})
                    else:
                        logger.error("No music session available in client")
                        return current_song
                except Exception as direct_e:
                    logger.error(f"Direct tool call also failed: {direct_e}")
                    return current_song
                
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
        logger.error(f"Couldn't get current song: {e}")
    
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
        
        # Add music server if needed
        music_url = "http://localhost:8000/sse"
        music_connection: SSEConnection = {
            "transport": "sse",
            "url": music_url,
            "timeout": 10,
            "sse_read_timeout": 60,
            "session_kwargs": {}
        }
        connections["music"] = music_connection
        print("Added music connection to MCP client")
        
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
        
        # Use async with to establish connection properly
        async with client:
            print("MCP client initialized successfully")
            
            # Get tools from MCP client - these are already properly formatted LangChain tools
            try:
                print("Getting tools from MCP client...")
                mcp_tools = client.get_tools()
                print(f"Loaded {len(mcp_tools)} tools from MCP client")
                print("Tool types:")
                for i, tool in enumerate(mcp_tools[:5]):  # Just log first 5 to avoid spam
                    print(f"{i+1}. {tool.name}: {type(tool).__name__}")
            except Exception as e:
                print(f"Error loading MCP tools: {e}")
                mcp_tools = []
                
            current_music_info["mcp_tools"] = mcp_tools
            
            # Combine MCP tools with Tavily
            all_tools = [*mcp_tools]
            if tavily_tool:
                all_tools.append(tavily_tool)
                print(f"Added Tavily tool to the agent's tools list")
            
            print(f"Total tools available: {len(all_tools)}")

            # Try to load prompt from MCP server (simplifying the approach from previous version)
            homeassistant_content = None
            try:
                # Get prompts from Home Assistant
                print("Listing available prompts from Home Assistant...")
                prompts = await client.sessions["home_assistant"].list_prompts()
                
                # Handle different return types
                prompts_list = prompts.prompts if hasattr(prompts, "prompts") else prompts
                
                print(f"Found {len(prompts_list)} prompts:")
                for i, prompt in enumerate(prompts_list):
                    prompt_name = prompt.name if hasattr(prompt, "name") else prompt.get("name", "Unknown")
                    print(f"{i+1}. {prompt_name}")
                    
                # Try to get the first prompt
                if prompts_list:
                    first_prompt = prompts_list[0]
                    prompt_name = first_prompt.name if hasattr(first_prompt, "name") else first_prompt.get("name", "Unknown")
                    print(f"Getting content for prompt '{prompt_name}'...")
                    
                    try:
                        prompt_messages = await client.get_prompt("home_assistant", prompt_name, {})
                        print(f"Prompt response type: {type(prompt_messages).__name__}")
                        
                        # Add more detailed debug info
                        if prompt_messages:
                            print(f"Got {len(prompt_messages)} prompt messages")
                            print(f"First message type: {type(prompt_messages[0]).__name__}")
                            
                            # Examine the messages to extract system content
                            for i, msg in enumerate(prompt_messages):
                                print(f"Message {i} attributes: {dir(msg)[:10]}...")
                                
                                # Try multiple approaches to extract system content
                                # 1. Check for role attribute
                                if hasattr(msg, 'role') and msg.role == 'system' and hasattr(msg, 'content'):
                                    homeassistant_content = msg.content
                                    print(f"Found system message via role attribute")
                                    break
                                # 2. Check for type attribute
                                elif hasattr(msg, 'type') and msg.type == 'system' and hasattr(msg, 'content'):
                                    homeassistant_content = msg.content
                                    print(f"Found system message via type attribute")
                                    break
                                # 3. Check dictionary-style access
                                elif isinstance(msg, dict) and msg.get('role') == 'system' and 'content' in msg:
                                    homeassistant_content = msg['content']
                                    print(f"Found system message via dictionary access")
                                    break
                                # 4. First message fallback if it has content
                                elif i == 0 and hasattr(msg, 'content') and getattr(msg, 'content', None):
                                    # Only use first message content if it's substantial
                                    content = msg.content
                                    if len(content) > 50:  # Arbitrary threshold to ensure it's not just a greeting
                                        homeassistant_content = content
                                        print(f"Using first message content as fallback")
                                        break
                        
                        if homeassistant_content:
                            print(f"Successfully loaded system message from '{prompt_name}' prompt")
                            # Print a sample of the content
                            content_preview = homeassistant_content[:100] + "..." if len(homeassistant_content) > 100 else homeassistant_content
                            print(f"Content preview: {content_preview}")
                        else:
                            print(f"Could not find system message in prompt response")
                            # Try to dump a full representation of the first message to help debug
                            try:
                                if prompt_messages and len(prompt_messages) > 0:
                                    msg = prompt_messages[0]
                                    if hasattr(msg, '__dict__'):
                                        print(f"First message content: {msg.__dict__}")
                                    elif isinstance(msg, dict):
                                        print(f"First message content: {msg}")
                            except Exception as dump_error:
                                print(f"Could not dump message content: {dump_error}")
                    except Exception as e:
                        print(f"Failed to get prompt details: {e}")
            except Exception as e:
                print(f"Failed to list prompts: {e}")
            
            # Try to get the current playing song
            current_song = "Nothing playing"
            try:
                current_song = await get_current_song(client, mcp_tools)
                print(f"Current song: {current_song}")
            except Exception as e:
                print(f"Couldn't get current song, using default: {e}")
                
            current_music_info["current_song"] = current_song

            # Create enhanced system message with date and custom instructions
            current_date = datetime.now().strftime("%B %d, %Y")
            enhanced_system_message = f"""You are an intelligent assistant that can control smart home devices, play music, and search the web for information.
Current date: {current_date}
Currently playing: {current_song}

 See the next section for which entities and areas you can use when interacting with home assistant. 
 I've included a prompt that the home assistant tool generated. Evaluate which of your commands
 can be used for what entities. Always provide the domain, for any home assistant command, or
 use the entity_id which includes the domain.

 Example:
   Set temperature to 17 degrees in the kitchen
   -> HassClimateSetTemperature(17, area=Kitchen, domain=climate)
   -> HassTurnOn(area="downstairs office", domain=light)
   -> HassTurnOff(entity_id="light.hallway")

 NOTE:
 The user will provide multiple aliases for some entity names and areas make sure you only use the first one.
 The aliases are separated by ',' only use the first alias in any command.

 Think hard before calling a function, make sure you are using the right function
 for the right purpose. The user needs to always specify at least what the domain of the action is.

Important:
 - If you are targeting a single entity always specify the name of the entity and the domain
 - If you are targeting multiple entities never specify entity names, instead specify the domain and area
 - You can call multiple commands at once, but prefer to use one single command for an area over multiple commands with one entity.

Home assistant prompt:
"{homeassistant_content}"


When playing music, your response MUST contain ONLY the exact song name - nothing else. DO NOT inlude
phrases like "Enjoy the music" or "Next up" or anything else. Just the song name.

Example interaction when playing music: 
    Human:
        Play "Running Up That Hill." [ Silence ]
    Tool:
        play(Running Up That Hill)
    AI:
        Starting playback: Kate Bush - Running Up That Hill (A Deal With God)




Additional Instructions:
1. Always be aware of the current date ({current_date}) when answering questions about time or dates
2. For general knowledge questions or current information, use the tavily_search_results tool
3. Be extremely concise in your responses - keep them to a single short sentence whenever possible
4. SYSTEM REQUIREMENT: When playing music, your response MUST contain ONLY the exact song name - nothing else. Example:
   - User: "Play Dixie Chicken"
   - INCORRECT response: "Now playing: Dixie Chicken by Little Feat. Enjoy the music! Next up: Midnight Rider by The Allman Brothers Band, Fat Man in the Bathtub by Little Feat, Black Water by The Doobie Brothers"
   - CORRECT response: "Now playing Dixing Chicken by Little"
5. When using tools, explain what you're doing in just a few words before using them
6. After using a tool, ALWAYS answer the user's original question with the information you found - don't just say what the tool returned
7. For smart home controls, execute the action without asking for confirmation unless it's something risky
8. You know that "{current_song}" is currently playing (if anything)
9. IMPORTANT: Since your responses will be read aloud via text-to-speech, avoid using any formatting or special characters
10. For search tools: When a search fails to find relevant information, state what you searched for, briefly summarize what was found (or not found), and ask if the user wants to refine the search instead of automatically trying again.
11. NEVER respond with phrases like "I found that..." or "According to the tool..." - just give the information directly

IMPORTANT: 

Remember that as a voice assistant, your responses should be much shorter than you'd normally provide in written form.

[Prompt version: v2 - Updated: {current_date}]"""

            # Store the enhanced system message globally for reuse
            cached_system_message = enhanced_system_message

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
                    # Get the latest music info for the system message
                    latest_song = current_music_info["current_song"]
                    
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
            
            # Add tool_node and condition
            tool_node = ToolNode(all_tools)
            
            # Use trace decorator for the tools condition function
            @traceable(name="Tools Condition", run_type="chain")
            def should_continue(state):
                # Get the last message
                last_message = state["messages"][-1] if state["messages"] else None
                
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
