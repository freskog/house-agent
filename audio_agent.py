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

# Import the agent modules
from agent import make_graph, AgentState
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

class AgentInterface:
    """Interface between the audio server and the agent"""
    
    def __init__(self):
        self.graph = None
        self.graph_ctx = None
        self.processing_lock = asyncio.Lock()
        self.last_response_time = 0
        self.cooldown_period = 1.0  # seconds to wait before processing a new request
        
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
    
    async def process_transcription(self, transcription: TranscriptionResult) -> str:
        """Process transcription through the agent"""
        # Apply cooldown to avoid rapid-fire requests
        current_time = time.time()
        if current_time - self.last_response_time < self.cooldown_period:
            time_to_wait = self.cooldown_period - (current_time - self.last_response_time)
            await asyncio.sleep(time_to_wait)
            
        # Use a lock to ensure only one request is processed at a time
        async with self.processing_lock:
            # Create input state with the transcription
            input_state = AgentState(messages=[HumanMessage(content=transcription.text)])
            
            response_text = ""
            # Process the input and collect response
            async for chunk in self.graph.astream(input_state, stream_mode=["messages", "values"]):
                try:
                    # Each chunk is a tuple (stream_type, data) when using multiple stream modes
                    if isinstance(chunk, tuple) and len(chunk) == 2:
                        stream_type, data = chunk
                        
                        # Handle message chunks (LLM token streaming)
                        if stream_type == "messages" and isinstance(data, tuple) and len(data) == 2:
                            message_chunk, metadata = data
                            
                            # Extract node name
                            node_name = metadata.get("langgraph_node", "")
                            
                            # Collect content from agent node
                            if node_name == "agent" and hasattr(message_chunk, 'content') and message_chunk.content:
                                response_text += message_chunk.content
                except Exception as e:
                    print(f"Error processing chunk: {e}")
            
            # Update last response time
            self.last_response_time = time.time()
            
            return response_text.strip() or "I'm sorry, I couldn't process that request."

    async def cleanup(self):
        """Clean up resources"""
        if self.graph_ctx is not None:
            try:
                await self.graph_ctx.__aexit__(None, None, None)
                self.graph = None
                self.graph_ctx = None
            except Exception as e:
                print(f"Error cleaning up agent: {e}")

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
    
    # Setup signal handlers
    loop = asyncio.get_running_loop()
    for s in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(
            s, lambda: asyncio.create_task(cleanup(server, agent_interface))
        )
    
    # Start server
    await server.start()
    print(f"Audio server started on ws://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    
    # Keep running until interrupted
    try:
        # This will run forever until the process is stopped
        await asyncio.Future()
    finally:
        await cleanup(server, agent_interface)
        
async def cleanup(server, agent_interface):
    """Cleanup resources"""
    print("Shutting down server...")
    await server.stop()
    await agent_interface.cleanup()
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    asyncio.get_event_loop().stop()

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