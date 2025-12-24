import os
import sys
from dotenv import load_dotenv

# Load env vars from .env file
load_dotenv()

# Ensure we can import the agno module
try:
    from agno import Agent, OpenAIChat, CohereChat
    print("MATCH: Successfully imported agno with CohereChat!")
except ImportError as e:
    print(f"ERROR: Failed to import agno: {e}")
    sys.exit(1)

try:
    print("Attempting to create Agent with Cohere...")
    
    # We expect COHERE_API_KEY to be loaded from .env
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("ERROR: COHERE_API_KEY not found in environment!")
        sys.exit(1)
        
    print(f"DEBUG: Found Cohere Key: {api_key[:5]}...")

    model = CohereChat(id="command-a-03-2025", api_key=api_key)
    agent = Agent(model=model, description="Test agent")
    print("MATCH: Cohere Agent created successfully!")
    
    print("Running agent against Cohere API...")
    try:
        response = agent.run("Hello from Rust! Please confirm you are Cohere.")
        print(f"MATCH: Agent Response: {response}")
    except Exception as e:
        print(f"ERROR: Agent run failed: {e}")
        sys.exit(1)
         
except Exception as e:
    print(f"ERROR: Verification failed with error: {e}")
    sys.exit(1)

print("Verification script finished.")
