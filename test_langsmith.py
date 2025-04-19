from langgraph_sdk import get_client
import asyncio
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

async def test_langsmith_integration():
    # Initialize the client with the correct URL
    client = get_client(url="http://127.0.0.1:2024")
    
    # Create a new thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    print(f"Created thread: {thread_id}")
    
    # Run the agent graph with a simple question
    input_data = {"messages": [{"role": "user", "content": "What's the weather like today?"}]}
    
    # Stream the response to see the output
    print("Running agent with input:", input_data)
    async for chunk in client.runs.stream(
        thread_id,
        "agent",  # The name of our graph in langgraph.json
        input=input_data,
        stream_mode="updates",
    ):
        print(f"Receiving chunk: {chunk.event}")
        print(chunk.data)
        print("\n")
    
    print("Test completed successfully. Check your LangSmith dashboard to see the traces.")

if __name__ == "__main__":
    asyncio.run(test_langsmith_integration()) 