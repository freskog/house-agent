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
            
        # Add the new user message to conversation history
        user_message = HumanMessage(content=transcription.text)
        self.conversation_history.append(user_message)
        
        # Keep only the last 10 messages to prevent the context from growing too large
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
            
        # Create input state with the full conversation history
        input_state = AgentState(messages=self.conversation_history)
        
        # Process the input and get response
        result = await self.graph.ainvoke(input_state)

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
        self.conversation_history.append(AIMessage(content=cleaned_response))
        
        # Split the response into sentences for streaming
        # This regex splits on sentence boundaries while preserving punctuation
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_response)
        
        # Filter out empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        if not sentences:
            return "I'm sorry, I couldn't process that request."
            
        print(f"Splitting response into {len(sentences)} sentences for streaming")
        
        # Process each sentence with TTS and send it to the client
        client_to_use = current_client or self.current_client
        
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
                
                # If the graph has reached its end node, hang up silently after sending the response
                if graph_has_ended:
                    # Double-check that the cleaned response doesn't end with a question mark
                    # as TTS cleaning might have modified the content
                    if not cleaned_response.strip().endswith('?'):
                        print("Hanging up silently as graph has reached the END node")
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
        
        # Skip the goodbye message entirely if reason is empty
        # This allows for silent hang-ups when requested
        goodbye_message = ""
        if reason:
            # Create a friendly goodbye message if one was provided
            goodbye_message = reason
            # Ensure the goodbye message is concise
            if len(goodbye_message) > 50:
                # If message is too long, truncate it and just keep the goodbye part
                goodbye_message = "Thanks for the conversation. Goodbye!"
            
            # Make sure the message ends with a clear "goodbye" so users know the call is ending
            if not any(term in goodbye_message.lower() for term in ["goodbye", "bye", "end", "hang up"]):
                goodbye_message += " Goodbye!"
            
        print(f"Attempting to hang up with message: '{goodbye_message if goodbye_message else '[silent hang up]'}'")
        
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
            
            if cmd.lower() == "exit" or cmd.lower() == "quit":
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