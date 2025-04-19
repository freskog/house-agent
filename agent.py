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

# Define structured output format for the agent
class AgentResponse(BaseModel):
    """Structured format for agent responses"""
    messages: List[AIMessage] = Field(description="Messages to add to the conversation")

# Define state
class AgentState(TypedDict):
    messages: Annotated[List[SystemMessage | HumanMessage | AIMessage | ToolMessage], add_messages]

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
        # Get tools from MCP client - these are already properly formatted LangChain tools
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

IMPORTANT: When using Home Assistant tools, use the following format:

For turning devices on/off:
- HassTurnOn(name="Living Room Light", area="Living Room") 
- HassTurnOff(name="Kitchen Light", area="Kitchen")

For light control:
- HassLightSet(name="Bedroom Light", brightness=50) - Set brightness to 50%
- HassLightSet(name="Living Room Light", color="blue") - Change light color

For climate control:
- HassClimateSetTemperature(name="Living Room Thermostat", temperature=72)

For media controls:
- HassMediaPause(name="Living Room Speaker") 
- HassMediaUnpause(name="Kitchen Speaker")
- HassSetVolume(name="Bedroom TV", volume_level=30)

For position controls (blinds, shades, etc.):
- HassSetPosition(name="Living Room Blinds", position=80) - 0 is closed, 100 is open

Remember that as a voice assistant, your responses should be much shorter than you'd normally provide in written form."""

        # Create the model with tool-calling enabled
        # Use ChatOpenAI directly with LangSmith tracing - it will automatically be traced
        llm = ChatOpenAI(
            streaming=True,
            temperature=0,
            model="gpt-4o-mini",
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
                updated_system_message = enhanced_system_message.replace(
                    f"Currently playing: {current_song}",
                    f"Currently playing: {latest_song}"
                )
                
                messages = [
                    SystemMessage(content=updated_system_message),
                    *messages
                ]

            # Normal processing
            response = llm.invoke(messages)
            
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
                print(f"Turn on tool schema: {turn_on_tool.args_schema.schema()}")

        yield graph

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
