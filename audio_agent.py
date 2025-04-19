#!/usr/bin/env python3
"""
Integration of audio server with the house agent.

This script starts the audio server and connects it to the house agent.
"""

import asyncio
import signal
import argparse
import os
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

# Import the agent modules
from agent import make_graph, AgentState
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

def clean_text_for_tts(text: str) -> str:
    """Clean and preprocess text to make it more suitable for TTS
    
    Removes markdown, special formatting, and other elements that 
    don't work well when read aloud. Also optimizes for concise responses.
    """
    # Check if this is a search verification response - if so, be more careful with cleaning
    is_search_verification = False
    if "search" in text.lower() and any(term in text.lower() for term in ["query", "found", "results", "refine"]):
        is_search_verification = True
        
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
    
    # If this is a search verification, don't remove these phrases
    if not is_search_verification:
        # Remove redundant confirmations and verbose phrases
        text = re.sub(r'(?i)I\'ll (help you|do that|get that for you|take care of that)', '', text)
        text = re.sub(r'(?i)(sure|certainly|of course|absolutely|definitely|no problem)', '', text)
        text = re.sub(r'(?i)I\'d be happy to', '', text)
        text = re.sub(r'(?i)Let me', '', text)
        text = re.sub(r'(?i)Here\'s', '', text)
        text = re.sub(r'(?i)For you', '', text)
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    # If we've removed too much, make sure there's still some text
    if not text:
        return "OK."
    
    return text

class AgentInterface:
    """Interface between the audio server and the agent"""
    
    def __init__(self):
        self.graph = None
        self.graph_ctx = None
        self.last_response_time = 0
        self.cooldown_period = 1.0  # seconds to wait before processing a new request
        self.is_speech_active = False
        self.speech_frames: List[bytes] = []
        self.last_speech_time = time.time()  # Initialize with current time to avoid NoneType error
        self.recording_start_time = time.time()  # Initialize with current time
        self.pre_vad_buffer = collections.deque(maxlen=5)  # Adjust buffer size as needed
        self.audio_server = None  # Reference to the audio server
        self.current_client = None  # Current websocket client
        self._last_client_address = None  # Track the last client address for debugging
        # Store conversation history
        self.conversation_history = []  # List to store all conversation messages
        self._loop_count = 0
        
    async def initialize(self):
        """Initialize the agent graph"""
        try:
            # Eagerly initialize the graph at startup
            print("Initializing agent graph...")
            self.graph_ctx = make_graph()
            self.graph = await self.graph_ctx.__aenter__()
            
            # Find and set the callback for the hang_up tool
            try:
                # Import the global hang_up_tool_instance
                from agent import hang_up_tool_instance
                
                if hang_up_tool_instance is not None:
                    # Set the callback directly on the instance
                    hang_up_tool_instance.set_callback(self.hang_up)
                    print("Hang Up tool callback connected successfully")
                else:
                    print("WARNING: Hang Up tool instance not found, hang-up functionality will not work")
            except ImportError:
                print("ERROR importing hang_up_tool_instance - hang-up functionality will not work")
            except Exception as e:
                print(f"ERROR setting hang up callback: {e}")
                import traceback
                traceback.print_exc()
            
            print("Agent interface initialized successfully")
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
    
    async def process_transcription(self, transcription: TranscriptionResult) -> str:
        """Process transcription through the agent"""
        # Store the current client from the websocket server context
        # Don't rely on the transcription.client as it's not serializable
        current_client = None
        if hasattr(transcription, '_websocket') and transcription._websocket:
            current_client = transcription._websocket
            # Update our client reference for hang-up functionality
            if current_client != self.current_client:
                self.current_client = current_client
                if hasattr(current_client, 'remote_address'):
                    self._last_client_address = current_client.remote_address
                    print(f"Updated client reference: {self._last_client_address}")
                else:
                    print("Updated client reference but no remote_address attribute")

        # Check if this is a hang-up command directly from the user
        if transcription.text and any(term in transcription.text.lower() for term in [
            "hang up", "end call", "disconnect", "goodbye", "bye", "end the call", "that's all"
        ]):
            print(f"Direct hang-up command detected: '{transcription.text}'")
            if self.current_client:
                # Hang up immediately without going through the agent
                await self.hang_up("Thanks for the conversation. Goodbye!")
                return ""
            
        # Add the new user message to conversation history
        user_message = HumanMessage(content=transcription.text)
        self.conversation_history.append(user_message)
        
        # Keep only the last 10 messages to prevent the context from growing too large
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
            
        # Create input state with the full conversation history
        input_state = AgentState(messages=self.conversation_history)
        
        response_text = ""
        last_chunk_time = time.time()
        # Track if we've gotten enough content to consider this a complete response
        has_sufficient_content = False
        should_end_conversation = False
        tool_result_processed = False  # Track if we've processed a tool result
        
        # Process the input and collect response
        async for chunk in self.graph.astream(input_state, stream_mode=["messages", "values"]):
            try:
                # Each chunk is a tuple (stream_type, data) when using multiple stream modes
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    stream_type, data = chunk
                    
                    # Check for the should_end_conversation flag in values stream
                    if stream_type == "values" and isinstance(data, dict):
                        if "should_end_conversation" in data and data["should_end_conversation"]:
                            print("DEBUG: Agent indicated conversation should end")
                            should_end_conversation = True
                    
                    # Handle message chunks (LLM token streaming)
                    if stream_type == "messages" and isinstance(data, tuple) and len(data) == 2:
                        message_chunk, metadata = data
                        
                        # Extract node name
                        node_name = metadata.get("langgraph_node", "")
                        
                        # Collect content from agent node
                        if node_name == "agent" and hasattr(message_chunk, 'content') and message_chunk.content:
                            # Check if this is likely a verification response (very short after a tool call)
                            # If tool verification responses are getting repeated, we should limit them
                            current_time = time.time()
                            new_content = message_chunk.content
                            
                            # Check if the response mentions verification terms we want to avoid
                            verification_terms = ["tool result", "tool returned", "i found", "the result shows", "according to the tool"]
                            contains_verification_terms = any(term in new_content.lower() for term in verification_terms)
                            
                            # Detect if this is a verification response about search results
                            is_search_verification = False
                            if "search" in new_content.lower() and any(term in new_content.lower() for term in ["query", "found", "results", "refine"]):
                                is_search_verification = True
                                # For search verifications, we want to include them fully
                                print("Detected search verification response")
                            
                            # Only add the content if it's not a duplicate of what we already have
                            # Always include search verification responses
                            # Skip responses that are just verifying the tool worked
                            if ((not response_text or not new_content.strip() in response_text) and 
                                (not contains_verification_terms or is_search_verification)):
                                response_text += new_content
                                last_chunk_time = current_time
                                
                                # Consider the response sufficient if we have a reasonable amount of content
                                if len(response_text) > 15:
                                    has_sufficient_content = True
                            elif current_time - last_chunk_time > 3.0 and has_sufficient_content:
                                # If we're getting duplicates and have sufficient content, stop processing
                                print("Detected potential response loop, stopping processing")
                                break
            except Exception as e:
                print(f"Error processing chunk: {e}")
        
        # Don't return the text - we'll process and stream it directly
        # Instead, return an empty string to indicate we're handling it ourselves
        if not response_text.strip():
            # If no response, return a fallback message
            return "I'm sorry, I couldn't process that request."
        
        # Check for the ##END## marker which indicates the conversation should end
        should_end_with_marker = "##END##" in response_text
        
        # Remove the ##END## marker from the response if present
        if should_end_with_marker:
            response_text = response_text.replace("##END##", "").strip()
            should_end_conversation = True
            print("Found ##END## marker - will end conversation after response")
        
        # Clean the response text to make it more suitable for TTS
        cleaned_response = clean_text_for_tts(response_text.strip())
        print(f"Original response length: {len(response_text)}, Cleaned response length: {len(cleaned_response)}")
        
        # Check if this was a weather or simple information query - if so, force end conversation
        weather_keywords = ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'climate', 'hot', 'cold']
        info_keywords = ['time', 'date', 'when is', 'what time', 'what day']
        
        if transcription.text and (
            any(keyword in transcription.text.lower() for keyword in weather_keywords) or
            any(keyword in transcription.text.lower() for keyword in info_keywords)
        ):
            print("Detected weather or time query - will end conversation after response")
            should_end_conversation = True
        
        # Add the agent's response to conversation history
        self.conversation_history.append(AIMessage(content=cleaned_response))
        
        # If the agent determined the conversation should end, hang up after delivering the message
        if should_end_conversation:
            print(f"Agent determined conversation should end: '{cleaned_response}'")
            # We will hang up after delivering the agent's response message
            
        # If the response seems to be repeating (potential loop), truncate it
        # Look for repeated phrases that might indicate a loop
        words = cleaned_response.split()
        detected_loop = False
        if len(words) > 10:
            # Check for repeating patterns
            for pattern_length in range(3, min(10, len(words) // 2)):
                for i in range(len(words) - pattern_length * 2):
                    pattern1 = ' '.join(words[i:i+pattern_length])
                    pattern2 = ' '.join(words[i+pattern_length:i+pattern_length*2])
                    if pattern1 == pattern2:
                        print(f"Detected repeating pattern: '{pattern1}', truncating response")
                        # Truncate to just before the repetition
                        cleaned_response = ' '.join(words[:i+pattern_length])
                        detected_loop = True
                        break
                if detected_loop:
                    break
        
        # If we detected a loop and we have a client, consider hanging up to prevent bad UX
        if detected_loop and self.audio_server and current_client:
            loop_count = getattr(self, '_loop_count', 0) + 1
            self._loop_count = loop_count
            
            # If we've detected multiple loops, hang up automatically
            if loop_count >= 2:
                print(f"Multiple response loops detected ({loop_count}), hanging up automatically")
                await self.hang_up("I seem to be having trouble. Let's end the call for now.")
                return ""
        
        # Check if the agent wants to hang up based on its response
        if any(term in cleaned_response.lower() for term in [
            "goodbye", "bye", "hang up", "end call", "end the call", "that's all"
        ]):
            print(f"Agent response indicates hang-up: '{cleaned_response}'")
            if self.current_client:
                # We already have the goodbye message in the response
                await self.hang_up(cleaned_response)
                return ""
        
        # Split the response into sentences for streaming
        # This regex splits on sentence boundaries while preserving punctuation
        import re
        # This pattern looks for sentence endings (., !, ?) followed by a space or end of string
        # It's careful not to split on periods in numbers, abbreviations, etc.
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_response)
        
        # Filter out empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return "I'm sorry, I couldn't process that request."
            
        print(f"Splitting response into {len(sentences)} sentences for streaming")
        
        # Use either the stored current client or the one from the instance
        client_to_use = current_client or self.current_client
        
        # Process each sentence with TTS and send it to the client
        if self.audio_server and client_to_use:
            try:
                # Process sentences in batches for more natural pauses
                batches = []
                current_batch = ""
                
                # Group sentences into reasonable batches (1-2 sentences per batch)
                # avoiding very short audio clips while preventing very long ones
                for sentence in sentences:
                    # If adding this sentence would make the batch too long, start a new batch
                    if len(current_batch) + len(sentence) > 100:  # Reduced character threshold for more concise responses
                        if current_batch:
                            batches.append(current_batch)
                            current_batch = sentence
                        else:
                            # If a single sentence is very long, keep it as its own batch
                            batches.append(sentence)
                    else:
                        # Add to current batch with a space if not empty
                        if current_batch:
                            current_batch += " " + sentence
                        else:
                            current_batch = sentence
                
                # Add the last batch if not empty
                if current_batch:
                    batches.append(current_batch)
                    
                print(f"Grouped into {len(batches)} audio batches")
                
                # Process each batch and stream to client
                for i, batch in enumerate(batches):
                    print(f"Processing batch {i+1}/{len(batches)}: {batch[:30]}...")
                    
                    # Skip if this is the same as the previous batch (to prevent repeats)
                    if i > 0 and batch == batches[i-1]:
                        print(f"Skipping duplicate batch: {batch[:30]}...")
                        continue
                    
                    # Use the TTS engine to synthesize speech
                    audio_data = await self.audio_server.tts_engine.synthesize(batch)
                    
                    if audio_data and len(audio_data) > 0:
                        # Send the audio data to client
                        await self.audio_server.send_audio_playback(client_to_use, audio_data)
                        print(f"Streamed audio batch {i+1} ({len(audio_data)} bytes)")
                        
                        # Small delay between batches for natural pausing
                        await asyncio.sleep(0.05)
                        
                        # If this is a very short response or the last batch, add a slightly longer pause
                        if len(batches) == 1 or i == len(batches) - 1:
                            await asyncio.sleep(0.5)  # Longer pause at the end
                    else:
                        print(f"Error: Empty audio data for batch {i+1}")
                
                print(f"Finished streaming all audio batches")
                
                # If the agent determined the conversation should end, hang up now
                if should_end_conversation:
                    print("Hanging up as agent determined conversation should end")
                    # Use the final response as the hang-up message
                    await self.hang_up(cleaned_response)
                    return ""
                
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
        
        # Create a friendly goodbye message if one wasn't provided
        goodbye_message = reason
        if not goodbye_message:
            goodbye_message = "Thanks for the conversation. Goodbye!"
        
        # Ensure the goodbye message is concise
        if len(goodbye_message) > 50:
            # If message is too long, truncate it and just keep the goodbye part
            goodbye_message = "Thanks for the conversation. Goodbye!"
        
        # Make sure the message ends with a clear "goodbye" so users know the call is ending
        if not any(term in goodbye_message.lower() for term in ["goodbye", "bye", "end", "hang up"]):
            goodbye_message += " Goodbye!"
            
        print(f"Attempting to hang up with message: '{goodbye_message}'")
        
        # First clear the client reference to prevent any further processing
        # Store it temporarily for the hang-up call
        temp_client = self.current_client
        self.current_client = None
        
        try:
            # Call the hang_up method on the audio server
            success = await self.audio_server.hang_up(
                websocket=temp_client,
                message=goodbye_message
            )
            
            if success:
                print(f"Hang up successful for client {self._last_client_address}")
                return "Call ended successfully"
            else:
                print(f"Hang up failed: Audio server reported failure for client {self._last_client_address}")
                # Restore client reference if hang-up failed
                self.current_client = temp_client
                return "Failed to hang up the call: Audio server error"
        except Exception as e:
            print(f"Hang up failed with exception: {e}")
            import traceback
            traceback.print_exc()
            # Restore client reference if hang-up failed
            self.current_client = temp_client
            return f"Failed to hang up the call: {e}"

async def test_hang_up(agent_interface):
    """Test the hang up functionality directly"""
    print("Testing hang up functionality...")
    
    if not agent_interface.current_client:
        print("No active client connected, cannot test hang up")
        return False
        
    result = await agent_interface.hang_up("This is a test of the hang up functionality. Goodbye!")
    print(f"Hang up test result: {result}")
    return "Call ended successfully" in result

async def main(args):
    """Main entry point"""
    # Create configs
    vad_config = VADConfig(
        threshold=args.vad_threshold,
        min_speech_duration=args.min_speech_duration,
        min_silence_duration=args.min_silence_duration,
        sample_rate=args.sample_rate
    )
    
    transcription_config = TranscriptionConfig(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language if args.language != "auto" else None
    )
    
    # Create and initialize agent interface
    agent_interface = AgentInterface()
    init_success = await agent_interface.initialize()
    if not init_success:
        print("Failed to initialize agent. Exiting.")
        return
    
    # Create server - this will eagerly initialize all components
    print(f"Starting audio server with {args.model} model on {args.device}...")
    server = AudioServer(
        host=args.host,
        port=args.port,
        vad_config=vad_config,
        transcription_config=transcription_config,
        transcription_callback=agent_interface.process_transcription
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
    print(f"Audio server started on ws://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    # Setup command processing
    async def process_command():
        """Process admin commands"""
        while True:
            cmd = await loop.run_in_executor(None, input, "\nAdmin command (or press Enter to skip): ")
            
            if cmd.lower() == "test_hangup":
                # Wait for a client to connect first
                if not agent_interface.current_client:
                    print("No client connected yet. Connect a client before testing hang up.")
                else:
                    # Test hang up functionality
                    await test_hang_up(agent_interface)
            elif cmd.lower() == "exit" or cmd.lower() == "quit":
                print("Exiting...")
                shutdown_event.set()
                break
            
            await asyncio.sleep(1.0)
    
    # Start command processing in background
    cmd_task = asyncio.create_task(process_command())
    
    # Monkey-patch the server's handle_connection method to capture the client
    original_handle_connection = server.handle_connection
    
    async def patched_handle_connection(websocket):
        # Store the client reference in the agent interface, but directly, not through serializable state
        agent_interface.current_client = websocket
        print(f"Client connected: {websocket.remote_address}")
        
        # Call the original method
        await original_handle_connection(websocket)
    
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
    parser = argparse.ArgumentParser(description="Audio Agent Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--vad-threshold", type=float, default=0.3, help="VAD threshold")
    parser.add_argument("--min-speech-duration", type=float, default=0.5, help="Minimum speech duration")
    parser.add_argument("--min-silence-duration", type=float, default=1.0, help="Minimum silence duration")
    parser.add_argument("--model", type=str, default="medium-q5_0", help="Whisper model to use")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run model on")
    parser.add_argument("--compute-type", type=str, default="int8", help="Compute type for model")
    parser.add_argument("--language", type=str, default="auto", help="Language for transcription")
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("Server stopped by user") 