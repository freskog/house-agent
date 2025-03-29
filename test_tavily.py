import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override = True)

# Get Tavily API key
tavily_api_key = os.getenv("TAVILY_API_KEY")
print(f"Using Tavily API Key: {tavily_api_key[:5]}...{tavily_api_key[-5:]}")

# Test Tavily API key directly with their API
try:
    # Define the API endpoint
    url = "https://api.tavily.com/search"
    
    # Set up the request headers and parameters
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": tavily_api_key
    }
    
    data = {
        "query": "What is the capital of France?",
        "search_depth": "basic",
        "max_results": 3,
        "include_answer": True,
        "include_images": False,
        "include_raw_content": False
    }
    
    # Make the request
    print("Testing Tavily API directly...")
    response = requests.post(url, headers=headers, json=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        print("✅ Success! Tavily API key is working correctly.")
        print(f"Response status: {response.status_code}")
        print(f"Response length: {len(response.text)} characters")
        print("\nSample of results:")
        print(response.text[:200] + "...")
    else:
        print(f"❌ Error: Received status code {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 401:
            print("\nThis is definitely an authentication error. Your API key is invalid or expired.")
            print("Please get a new API key from: https://tavily.com/")
            print("\nThe key should look something like: tvly-ABCDEFghijklMNOPqrstUVWXyz12345678")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPossible issues:")
    print("1. The API key may be invalid or expired")
    print("2. There might be network connectivity issues")
    print("3. The Tavily service might be experiencing problems") 