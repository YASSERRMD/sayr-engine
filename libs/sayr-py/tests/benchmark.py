import time
import os
import cohere
from agno import Agent, CohereChat
from dotenv import load_dotenv

# Load env vars
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    print("Error: COHERE_API_KEY not found")
    exit(1)

MODEL_ID = "command-a-03-2025"
ITERATIONS = 10

def benchmark_pure_python():
    print(f"\n--- Testing Pure Python (cohere sdk) ---")
    co = cohere.ClientV2(api_key)
    latencies = []
    
    for i in range(ITERATIONS):
        print(f"Iteration {i+1}/{ITERATIONS}...", end="", flush=True)
        start = time.time()
        try:
            # Matches the simple chat interface
            response = co.chat(
                model=MODEL_ID, 
                messages=[{"role": "user", "content": "Say hello in one word."}]
            )
            # Accessing content to ensure full processing
            _ = response.message.content[0].text
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        end = time.time()
        duration = end - start
        latencies.append(duration)
        print(f" {duration:.4f}s")
        
    avg = sum(latencies) / len(latencies) if latencies else 0
    print(f"Pure Python Average: {avg:.4f}s")
    return avg

def benchmark_rust_bindings():
    print(f"\n--- Testing Agno Rust Bindings ---")
    # Re-instantiate per test or reuse? Usually we reuse the agent.
    model = CohereChat(id=MODEL_ID, api_key=api_key)
    agent = Agent(model=model, description="Bench agent", markdown=False)
    latencies = []

    for i in range(ITERATIONS):
        print(f"Iteration {i+1}/{ITERATIONS}...", end="", flush=True)
        start = time.time()
        try:
            _ = agent.run("Say hello in one word.")
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        end = time.time()
        duration = end - start
        latencies.append(duration)
        print(f" {duration:.4f}s")

    avg = sum(latencies) / len(latencies) if latencies else 0
    print(f"Rust Bindings Average: {avg:.4f}s")
    return avg


# -----------------------------------------------------------------------------
# CPU Benchmarks
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# CPU Benchmarks
# -----------------------------------------------------------------------------

def py_fibonacci(n):
    if n <= 1:
        return n
    return py_fibonacci(n-1) + py_fibonacci(n-2)

def benchmark_fibonacci():
    from agno import fibonacci as rust_fibonacci
    print("\n--- Benchmarking CPU: Recursive Fibonacci(30) ---")
    
    # Python
    start = time.time()
    for _ in range(ITERATIONS):
        py_fibonacci(30)
    end = time.time()
    py_avg = (end - start) / ITERATIONS
    print(f"1. Agno Python (Native):      {py_avg:.6f}s")
    
    # Rust Bindings
    start = time.time()
    for _ in range(ITERATIONS):
        rust_fibonacci(30)
    end = time.time()
    rust_avg = (end - start) / ITERATIONS
    print(f"2. Agno-Rust (Python+Rust):   {rust_avg:.6f}s")
    
    return py_avg, rust_avg

def benchmark_token_count():
    from agno import calculate_tokens as rust_tokens
    print("\n--- Benchmarking CPU: Token Count (10MB String) ---")
    
    # Generate 10MB string
    large_text = "word " * 2_000_000
    
    # Python
    start = time.time()
    for _ in range(ITERATIONS):
        len(large_text.split())
    end = time.time()
    py_avg = (end - start) / ITERATIONS
    print(f"1. Agno Python (Native):      {py_avg:.6f}s")
    
    # Rust Bindings
    start = time.time()
    for _ in range(ITERATIONS):
        rust_tokens(large_text)
    end = time.time()
    rust_avg = (end - start) / ITERATIONS
    print(f"2. Agno-Rust (Python+Rust):   {rust_avg:.6f}s")

    return py_avg, rust_avg

if __name__ == "__main__":
    print(f"Starting CPU Benchmark (Iterations: {ITERATIONS})")
    benchmark_fibonacci()
    benchmark_token_count()

