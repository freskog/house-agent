#!/usr/bin/env python3
"""
Integration of audio server with the house agent.

This script starts the audio server and connects it to the house agent.
"""

import asyncio
import signal
import argparse
import os
import locale
from audio.server import (
    AudioServer, 
    VADConfig, 
    TranscriptionConfig,
    TranscriptionResult
)
from dotenv import load_dotenv
import time
from typing import List
import collections
import re
import sys
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable

# Set C locale to avoid Whisper segmentation fault on systems with non-C locales
try:
    print("Setting C locale to prevent Whisper segfault...")
    locale.setlocale(locale.LC_ALL, 'C')
    os.environ['LC_ALL'] = 'C'
    os.environ['LANG'] = 'C'
    os.environ['LANGUAGE'] = 'C'
    print("Locale set to C successfully")
except Exception as e:
    print(f"Warning: Could not set locale to C: {e}")

# Import the agent modules
from agent import make_graph, AgentState

# Load environment variables
load_dotenv()    

def clean_text_for_tts(text: str) -> str:
    """Clean and preprocess text to make it more suitable for TTS
    
    Removes markdown, special formatting, and other elements that 
    don't work well when read aloud. Also optimizes for concise responses.
    """
    # Replace markdown code blocks (both with and without language specification)
    text = re.sub(r'```[\w]*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
    
    # Replace markdown bullet points with proper speech pauses
    text = re.sub(r'^\s*[-*+]\s+', '. ', text, flags=re.MULTILINE)
    
    # Replace markdown headers
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1.', text, flags=re.MULTILINE)
    
    # Replace excessive newlines with a single period for a pause
    text = re.sub(r'\n{2,}', '. ', text)
    # Replace single newlines with spaces
    text = re.sub(r'\n', ' ', text)
    
    # Remove markdown links and just keep the text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Replace markdown emphasis/bold with plain text
    text = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', text)
    
    # Remove any remaining special characters that aren't good for TTS
    text = re.sub(r'[#*_~`|]', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    # Fix cases where punctuation might not have spaces after it
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    
    # Add final period if not present
    if text and text[-1] not in ['.', '!', '?']:
        text += '.'
    
    # Remove redundant confirmations and verbose phrases - expanded list
    verbose_phrases = [
        # Confirmations and acknowledgments
        r'(?i)I\'ll (help you|do that|get that for you|take care of that|assist you with that)',
        r'(?i)(sure|certainly|of course|absolutely|definitely|no problem)',
        r'(?i)I\'d be (happy|glad) to',
        # Unnecessary phrase beginnings
        r'(?i)Let me',
        r'(?i)I can',
        r'(?i)I will',
        r'(?i)Here\'s',
        r'(?i)For you',
        r'(?i)I think',
        # Hedging phrases
        r'(?i)It (seems|appears|looks) like',
        r'(?i)Based on (the|your)',
        r'(?i)According to',
        # Self-references
        r'(?i)As an (AI|assistant)',
        r'(?i)As your (AI|assistant|helper)',
        # Transition phrases
        r'(?i)Now, ',
        r'(?i)So, ',
        # Redundant instructions 
        r'(?i)You (can|could|should|might want to)',
        r'(?i)I (suggest|recommend)',
    ]
    
    # Apply all the verbose phrase removals
    for phrase in verbose_phrases:
        text = re.sub(phrase, '', text)
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    # Remove redundant punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'[.!?][.!?]+', '.', text)
    
    # Trim spaces before punctuation
    text = re.sub(r'\s+([.!?,;:])', r'\1', text)
    
    # Remove double spaces that may have been created during cleanup
    text = re.sub(r'\s{2,}', ' ', text)
    
    # If we've removed too much, make sure there's still some text
    if not text:
        return ""  # Return empty string for silent responses instead of "OK."
    
    return text

class AgentInterface:
    """Interface between the audio server and the agent"""
    
    def __init__(self):
        self.graph = None
        self.graph_ctx = None
        self.last_response_time = 0
        self.cooldown_period = 0.5  # seconds to wait before processing a new request (reduced from 1.0s)
        self.is_speech_active = False
        self.speech_frames: List[bytes] = []
        self.last_speech_time = time.time()  # Initialize with current time to avoid NoneType error
        self.recording_start_time = time.time()  # Initialize with current time
        self.pre_vad_buffer = collections.deque(maxlen=5)  # Adjust buffer size as needed
        self.audio_server = None  # Reference to the audio server
        self.current_client = None  # Current websocket client
        self._last_client_address = None  # Track the last client address for debugging
        # Store conversation history per client
        self.client_conversation_history = {}  # Dictionary to store conversation history per client
        self._loop_count = 0
        self._is_agent_initialized = False  # Track if the agent is fully initialized
        
    async def initialize(self):
        """Initialize the agent graph"""
        try:
            # Eagerly initialize the graph at startup
            print("Initializing agent graph...")
            start_time = time.time()
            self.graph_ctx = make_graph()
            
            # Use a try-except block to handle the initialization
            try:
                self.graph = await self.graph_ctx.__aenter__()
            except TypeError as e:
                # If there's a TypeError, it might be related to cache_seed or other parameters
                # being passed incorrectly. Let's log it but continue
                print(f"Warning: Agent initialization had TypeError: {e}")
                print("Continuing with initialization despite the warning...")
                
            # Mark initialization as complete
            self._is_agent_initialized = True

            elapsed = time.time() - start_time
            print(f"Agent interface initialized successfully in {elapsed:.2f} seconds")
            return True
        except Exception as e:
            print(f"Error initializing agent: {e}")
            return False
            
    def set_audio_server(self, server, client=None):
        """Set the audio server reference and optionally the current client"""
        self.audio_server = server
        if client:
            self.current_client = client
            if hasattr(client, 'remote_address'):
                self._last_client_address = client.remote_address
                print(f"Client reference set: {self._last_client_address}")
            else:
                print("Client reference set but no remote_address attribute")
    
    @traceable(run_type="chain", name="Agent_Process_Transcription")
    async def process_transcription(self, transcription: TranscriptionResult) -> str:
        """Process transcription through the agent"""
        # Store the current client from the websocket server context
        # Don't rely on the transcription.client as it's not serializable
        current_client = None
        thread_id = None
        
        if hasattr(transcription, '_websocket') and transcription._websocket:
            current_client = transcription._websocket
            
            # Try to get the thread_id from the client_state
            if self.audio_server and hasattr(self.audio_server, 'client_states'):
                client_state = self.audio_server.client_states.get(current_client)
                if client_state and hasattr(client_state, 'thread_id'):
                    thread_id = client_state.thread_id
                    print(f"Using thread_id from client state: {thread_id}")
            
            # Update our client reference for hang-up functionality
            if current_client != self.current_client:
                self.current_client = current_client
                if hasattr(current_client, 'remote_address'):
                    self._last_client_address = current_client.remote_address
                    print(f"Updated client reference: {self._last_client_address}")
                else:
                    print("Updated client reference but no remote_address attribute")
        
        # Send an immediate acknowledgement tone if audio server is available
        client_to_use = current_client or self.current_client
        if self.audio_server and client_to_use:
            try:
                # You can create a very short "processing" sound or just a subtle tone
                # This creates perception of immediate responsiveness
                await self.audio_server.send_processing_indicator(client_to_use)
            except Exception as e:
                print(f"Error sending processing indicator: {e}")
            
        # Add the new user message to conversation history
        user_message = HumanMessage(content=transcription.text)
        if current_client not in self.client_conversation_history:
            self.client_conversation_history[current_client] = []
        self.client_conversation_history[current_client].append(user_message)
        
        # Keep only the last 10 messages to prevent context from growing too large and slowing down processing

        if len(self.client_conversation_history[current_client]) > 10:
            self.client_conversation_history[current_client] = self.client_conversation_history[current_client][-4:]
            
        # Create input state with the conversation history
        input_state = AgentState(
            messages=self.client_conversation_history[current_client],
            audio_server=self.audio_server,
            current_client=current_client
        )
        
        # Process the input and get response - measure response time
        start_time = time.time()
        
        # Pass thread_id to ainvoke if available
        if thread_id:
            print(f"Invoking graph with thread_id: {thread_id}")
            result = await self.graph.ainvoke(input_state, config={"thread_id": thread_id})
        else:
            print("Invoking graph without thread_id")
            result = await self.graph.ainvoke(input_state)
            
        elapsed = time.time() - start_time
        print(f"Agent processing completed in {elapsed:.2f} seconds")

        # Check if the graph has ended (reached the END node)
        # This is determined by examining the result structure
        graph_has_ended = False
        if result and "messages" in result and result["messages"] and len(result["messages"]) > 0:
            # If the graph has reached the END node, the last message is the final response
            # We'll detect this by checking if there are no more actions to take (no tool calls)
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                # If the message has content but no tool_calls, it's likely the end
                if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
                    # Only mark as ended if the response doesn't end with a question
                    # This ensures we don't hang up when the agent asks a question
                    response_content = last_message.content.strip()
                    if not response_content.endswith('?'):
                        graph_has_ended = True
                        print("Graph has reached the END node - will hang up silently after response")
                    else:
                        print("Graph has reached END node but response ends with a question - will NOT hang up")

        # Extract the text response from structured output
        response_text = ""

        if result and "messages" in result and result["messages"] and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                response_text = last_message.content

        # Clean the response for TTS
        cleaned_response = clean_text_for_tts(response_text.strip())
        print(f"Original response length: {len(response_text)}, Cleaned response length: {len(cleaned_response)}")
        
        # Add the agent's response to conversation history
        if current_client not in self.client_conversation_history:
            self.client_conversation_history[current_client] = []
        self.client_conversation_history[current_client].append(AIMessage(content=cleaned_response))
        
        # Check if this is an intentionally silent response (from silent_end node for music commands)
        # Only treat as silent if it's a music control command that succeeded
        is_silent_music_response = False
        if graph_has_ended and not cleaned_response.strip():
            print(f"DEBUG: Checking for silent music response. Messages count: {len(result['messages']) if result and 'messages' in result else 0}")
            # Check if the last AI message before this one had a music control tool call
            if result and "messages" in result and len(result["messages"]) >= 2:
                print("DEBUG: Searching for music control tool calls in message history")
                # Look for the second-to-last message (should be an AI message with tool calls)
                for i, msg in enumerate(reversed(result["messages"][:-1])):  # Skip the last (current) message
                    print(f"DEBUG: Message {i}: type={type(msg).__name__}, has_tool_calls={hasattr(msg, 'tool_calls')}")
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"DEBUG: Found message with {len(msg.tool_calls)} tool calls")
                        for tool_call in msg.tool_calls:
                            # Handle both attribute and dictionary access for tool_call name
                            tool_name = ''
                            if hasattr(tool_call, 'name'):
                                tool_name = tool_call.name
                            elif isinstance(tool_call, dict) and 'name' in tool_call:
                                tool_name = tool_call['name']
                            elif hasattr(tool_call, 'get'):
                                tool_name = tool_call.get('name', '')
                            
                            print(f"DEBUG: Tool call: {tool_name} (type: {type(tool_call).__name__})")
                            if tool_name in ["play_music", "pause_music", "stop_music", "next_track", "previous_track", "set_volume"]:
                                print(f"Detected successful music control command '{tool_name}' with empty response - this is intentionally silent")
                                is_silent_music_response = True
                                break
                        if is_silent_music_response:
                            break
        
        if is_silent_music_response:
            print("Silent music command response - hanging up without TTS")
            await self.hang_up("")  # Empty string means no goodbye message
            return ""
        
        # Check for empty response before any TTS processing
        if not cleaned_response.strip():
            print("Empty response - hanging up without TTS")
            await self.hang_up("")  # Empty string means no goodbye message
            return ""
        
        # Split the response into sentences for streaming
        # This regex splits on sentence boundaries while preserving punctuation
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_response)
        
        # Filter out empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            # Don't generate a fallback message for intentionally empty responses
            # This prevents TTS from speaking when the agent returns an empty response
            print("No sentences to process - hanging up without TTS")
            await self.hang_up("")  # Empty string means no goodbye message
            return ""
            
        print(f"Agent response has {len(sentences)} sentences for TTS")
        
        # Process each sentence with TTS and send it to the client
        client_to_use = current_client or self.current_client
        
        if self.audio_server and client_to_use:
            try:
                # Use the audio server's streaming TTS feature directly
                print(f"Using audio server streaming TTS for response")
                start_time = time.time()
                
                # Stream the cleaned response directly through the server's streaming TTS
                success = await self.audio_server.text_to_speech_streaming(cleaned_response, client_to_use)
                
                if success:
                    total_time = time.time() - start_time
                    print(f"TTS streaming completed in {total_time:.2f}s")
                else:
                    print(f"TTS streaming failed, falling back to server's non-streaming TTS")
                    # Let the server handle the fallback to non-streaming TTS
                    audio_data = await self.audio_server.text_to_speech(cleaned_response)
                    if audio_data:
                        await self.audio_server.send_audio_playback(client_to_use, audio_data)
                    else:
                        print("Both streaming and non-streaming TTS failed")
                
                # If the graph has reached its end node, hang up silently after sending the response
                # NOTE: Silent responses (empty content) are now handled earlier in the function
                # This section only handles responses that have content but still need hang up
                if graph_has_ended and cleaned_response.strip():
                    # Double-check that the cleaned response doesn't end with a question mark
                    # as TTS cleaning might have modified the content
                    if not cleaned_response.strip().endswith('?'):
                        print("Hanging up silently as graph has reached the END node (after response)")
                        await self.hang_up("")  # Empty string means no goodbye message
                        return ""
                    else:
                        print("Not hanging up as final response ends with a question")
                                
            except Exception as e:
                print(f"Error streaming TTS: {e}")
                import traceback
                traceback.print_exc()
                return "Error generating speech."
                
            # Return empty string since we've handled the audio streaming ourselves
            return ""
        else:
            # If no audio server, just return the text
            print("No audio server reference or client, returning full response")
            return cleaned_response

    async def cleanup(self):
        """Clean up resources"""
        if self.graph_ctx is not None:
            try:
                await self.graph_ctx.__aexit__(None, None, None)
                self.graph = None
                self.graph_ctx = None
            except Exception as e:
                print(f"Error cleaning up agent: {e}")
                
    async def hang_up(self, reason: str = "") -> str:
        """Hang up the call with the current client
        
        Args:
            reason: The reason for hanging up (will be spoken to the user)
                   If empty, no message will be played before hanging up
            
        Returns:
            str: A message indicating success or failure
        """
        print(f"Hang up requested with reason: '{reason}'")
        
        if not self.audio_server:
            print("No audio server available")
            return "Failed to hang up: No audio server available"
            
        client_to_use = self.current_client
        if not client_to_use:
            print(f"No active client connection (last known client: {self._last_client_address})")
            return "Failed to hang up: No active client connection"
            
        # Use the audio server's hang_up method which now supports streaming TTS
        try:
            success = await self.audio_server.hang_up(client_to_use, reason)
            if success:
                print(f"Hang up successful for client {self._last_client_address}")
                # Clear our client reference after successful hang up
                self.current_client = None
                return "Call ended successfully"
            else:
                print(f"Hang up failed: Audio server reported failure")
                return "Failed to hang up the call: Audio server error"
        except Exception as e:
            print(f"Hang up failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return f"Failed to hang up the call: {e}"

    async def cleanup_client(self, client):
        """Clean up resources for a specific client"""
        if client in self.client_conversation_history:
            print(f"Cleaning up conversation history for client {getattr(client, 'remote_address', 'unknown')}")
            del self.client_conversation_history[client]
            
        # If this was the current client, clear that reference
        if client == self.current_client:
            self.current_client = None
            self._last_client_address = None

async def main(args):
    """Main entry point"""
    try:
        print(f"Starting Audio Agent v2.0 (Optimized for low latency)")
        start_time = time.time()
        
        # Create VAD config with custom threshold
        vad_config = VADConfig(
            threshold=args.vad_threshold,
            min_speech_duration=args.min_speech_duration,
            min_silence_duration=args.min_silence_duration,
            sample_rate=args.sample_rate,
            window_size_samples=1536,  # Increased from default for better context
            buffer_size=10  # Number of frames to consider together
        )
        
        # Create transcription config
        transcription_config = TranscriptionConfig(
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            language=args.language if args.language != "auto" else None,
            use_coreml=args.use_coreml  # Use CoreML on Apple Silicon
        )
        
        # Create and initialize agent interface
        print("\n1. Initializing agent interface...")
        agent_interface = AgentInterface()
        init_success = await agent_interface.initialize()
        if not init_success:
            print("Failed to initialize agent. Exiting.")
            return
        
        # Create server - this will eagerly initialize all components
        print(f"\n2. Starting audio server with {args.model} model on {args.device}...")
        server = AudioServer(
            host=args.host,
            port=args.port,
            vad_config=vad_config,
            transcription_config=transcription_config,
            transcription_callback=agent_interface.process_transcription,
            save_recordings=args.save_recordings
        )
        
        # Set the server reference in the agent interface
        agent_interface.set_audio_server(server)
        
        # Create a shutdown event
        shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        loop = asyncio.get_running_loop()
        
        def signal_handler():
            print("Shutdown signal received")
            shutdown_event.set()
        
        for s in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(s, signal_handler)
        
        # Start server
        await server.start()
        
        total_startup_time = time.time() - start_time
        print(f"\nAudio agent ready! Startup completed in {total_startup_time:.2f} seconds")
        print(f"Server running on ws://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        
        # Setup command processing
        async def process_command():
            """Process admin commands"""
            while True:
                cmd = await loop.run_in_executor(None, input, "\nAdmin command (or press Enter to skip): ")
                
                if cmd.lower() == "exit" or cmd.lower() == "quit":
                    print("Exiting...")
                    shutdown_event.set()
                    break
                elif cmd.lower() == "status":
                    print("\nStatus:")
                    print(f"- Audio server running: yes")
                    print(f"- Agent initialized: {agent_interface._is_agent_initialized}")
                    print(f"- Current client: {agent_interface._last_client_address or 'None'}")
                    print(f"- Conversation history size: {len(agent_interface.client_conversation_history.get(agent_interface.current_client, [])) if agent_interface.current_client else 'None'} messages")
                
                await asyncio.sleep(0.1)  # Reduced sleep time
        
        # Start command processing in background
        cmd_task = asyncio.create_task(process_command())
        
        # Monkey-patch the server's handle_connection method to capture the client
        original_handle_connection = server.handle_connection
        
        async def patched_handle_connection(websocket):
            # Store the client reference in the agent interface, but directly, not through serializable state
            agent_interface.current_client = websocket
            print(f"Client connected: {websocket.remote_address}")
            
            try:
                # Call the original method
                await original_handle_connection(websocket)
            finally:
                # Clean up client resources when connection ends
                print(f"Client disconnected: {websocket.remote_address}")
                await agent_interface.cleanup_client(websocket)
        
        # Replace the method
        server.handle_connection = patched_handle_connection
        
        # Keep running until interrupted
        try:
            # Wait for shutdown event
            await shutdown_event.wait()
        finally:
            # Cancel the command task
            if not cmd_task.done():
                cmd_task.cancel()
            
            await cleanup(server, agent_interface)
        
    except Exception as e:
        print(f"Error starting audio agent: {e}")
        sys.exit(1)

async def cleanup(server, agent_interface):
    """Cleanup resources"""
    print("Shutting down server...")
    try:
        await server.stop()
    except Exception as e:
        print(f"Error stopping server: {e}")
    
    try:
        await agent_interface.cleanup()
    except Exception as e:
        print(f"Error cleaning up agent: {e}")
    
    # Be more careful with task cancellation to avoid recursion
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    # Cancel tasks safely - limit recursion
    for task in tasks:
        try:
            if not task.done():
                task.cancel()
        except Exception as e:
            print(f"Error cancelling task: {e}")
    
    # Wait for tasks with a timeout
    try:
        # Use wait_for with a timeout to prevent hanging
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        print("Some tasks did not complete within timeout")
    except Exception as e:
        print(f"Error waiting for tasks to complete: {e}")
    
    try:
        asyncio.get_event_loop().stop()
    except Exception as e:
        print(f"Error stopping event loop: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Audio Agent Server (Optimized for low latency)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--vad-threshold", type=float, default=0.15, help="VAD threshold (lower is more sensitive, default: 0.15)")
    parser.add_argument("--min-speech-duration", type=float, default=0.3, help="Minimum speech duration (default: 0.3s)")
    parser.add_argument("--min-silence-duration", type=float, default=0.8, help="Minimum silence duration (default: 0.8s)")
    parser.add_argument("--model", type=str, default="medium-q5_0", help="Whisper model to use (default: medium-q5_0)")
    parser.add_argument("--device", type=str, default="mps", help="Device to run model on (cpu/mps)")
    parser.add_argument("--compute-type", type=str, default="int8", help="Compute type for model")
    parser.add_argument("--language", type=str, default="auto", help="Language for transcription")
    parser.add_argument("--use-coreml", action="store_true", help="Use CoreML on Apple Silicon")
    parser.add_argument("--save-recordings", action="store_true", help="Save recordings")
    parser.add_argument("--use-langsmith", action="store_true", help="Enable LangSmith tracing (disabled by default)")
    parser.add_argument("--langsmith-api-key", type=str, help="LangSmith API key for tracing (starts with 'ls-')")
    parser.add_argument("--langsmith-project", type=str, help="LangSmith project for organizing traces")
    args = parser.parse_args()
    
    # Process the command line API key if provided
    if args.use_langsmith:
        if args.langsmith_api_key:
            os.environ["LANGSMITH_API_KEY"] = args.langsmith_api_key
            os.environ["LANGCHAIN_API_KEY"] = args.langsmith_api_key  # For compatibility
            
        if args.langsmith_project:
            os.environ["LANGSMITH_PROJECT"] = args.langsmith_project
            os.environ["LANGCHAIN_PROJECT"] = args.langsmith_project  # For compatibility
    else:
        # Disable LangSmith if not explicitly enabled
        os.environ["LANGSMITH_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    try:
        print("Starting Audio Agent with optimized latency settings...")
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("Server stopped by user") 