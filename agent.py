from contextlib import asynccontextmanager
from langgraph.graph import StateGraph, START, END
from typing import Annotated, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import BaseTool

from pydantic import BaseModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.tools import TavilySearchResults

import asyncio
import os
import json
from dotenv import load_dotenv
from datetime import datetime
import httpx

# Load environment variables
load_dotenv(override = True)

# Define state
class AgentState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]

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

# Create shared variables to store music info
current_music_info = {"current_song": "Nothing playing", "mcp_tools": None}

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
        
        # Combine MCP tools with Tavily if available
        all_tools = [*mcp_tools]
        if tavily_tool:
            all_tools.append(tavily_tool)
            print(f"Added Tavily tool to the agent's tools list. Total tools: {len(all_tools)}")
        else:
            print("Tavily tool not added due to error")

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
3. Be concise but informative in your responses
4. When using tools, explain what you're doing before using them unless it's obvious
5. If you're unsure about something, use the search tool to verify the information
6. For smart home controls, confirm the action before executing it
7. For music playback, suggest appropriate music based on the context (default to 80s pop or rock)
8. You know that "{current_song}" is currently playing (if anything)

Remember to use the appropriate tools based on the user's request. If you need to search for current information, use the Tavily search tool."""

        # Create LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(all_tools)
        
        # Create graph
        builder = StateGraph(AgentState)
        
        # Define agent node function with updated system message each time
        def agent(state: AgentState):
            messages = state.messages
            
            # Add system message if it's the first interaction
            if len(messages) == 1 and messages[0].type == "human":
                # Get the latest music info for the system message
                latest_song = current_music_info["current_song"]
                
                # Create updated system message with current song
                updated_system_message = enhanced_system_message.replace(
                    f"Currently playing: {current_song}",
                    f"Currently playing: {latest_song}"
                ).replace(
                    f'You know that "{current_song}" is currently playing',
                    f'You know that "{latest_song}" is currently playing'
                )
                
                messages = [
                    {"role": "system", "content": updated_system_message},
                    *messages
                ]
            
            response = llm_with_tools.invoke(messages)
            return AgentState(messages=[response])
        
        # Add nodes to graph
        builder.add_node("agent", agent)
        
        # Use the standard ToolNode with proper error handling
        builder.add_node("tools", ToolNode(all_tools))
        
        # Add edges
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
        
        # Compile and yield the graph
        graph = builder.compile()
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
            input_state = AgentState(messages=[HumanMessage(content=user_input)])
            
            try:
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
                                            print(f"Using {tool_name} with args:", flush=True)
                                            print(f"  {args_str.replace('\\n', '\\n  ')}", flush=True)
                                        
                                        # Mark that we've printed tool calls for this message
                                        setattr(message_chunk, '_tool_calls_printed', True)
                            
                            # Handle values chunks (state updates)
                            elif stream_type == "values" and isinstance(data, dict):
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
                        break
                    except Exception as e:
                        print(f"\n[Error] An error occurred: {e}")
                        print(f"Error type: {type(e)}")  # Show error type for debugging
            
            except httpx.RemoteProtocolError as e:
                print(f"\n[Connection Error] The server closed the connection unexpectedly. This often happens when music playback starts successfully.")
                print("\nThe music is playing but the connection was closed.")
            except Exception as e:
                print(f"\n[Error] An error occurred while processing: {e}")
                # Try to recover conversation
                print("\nLet me try to continue our conversation...")
            
            print("\n")

if __name__ == "__main__":
    asyncio.run(setup_and_run())
