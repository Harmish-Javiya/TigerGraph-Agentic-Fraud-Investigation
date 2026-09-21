import subprocess, sys, time, os

# Start Ollama server (detached)
print("Starting Ollama server...")
proc = subprocess.Popen(
    [r"C:\Users\123\AppData\Local\Programs\Ollama\ollama.exe", "serve"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    cwd=r"C:\Users\123\AppData\Local\Programs\Ollama",
    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
)
print(f"Ollama PID: {proc.pid}")

# Wait for server to be ready
print("Waiting for Ollama API...")
for i in range(30):
    time.sleep(1)
    try:
        r = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/tags"],
            capture_output=True, text=True, timeout=2
        )
        if r.returncode == 0:
            print(f"Ollama ready!")
            break
    except:
        pass
    if i % 5 == 0:
        print(f"  ...waiting ({i}s)")

# Pull the model (blocks until done)
print("\nPulling DeepSeek-R1-Distill-Qwen-7B model...")
pull = subprocess.run(
    [r"C:\Users\123\AppData\Local\Programs\Ollama\ollama.exe", "pull", "deepseek-r1-distill-qwen-7b"],
    cwd=r"C:\Users\123\AppData\Local\Programs\Ollama",
    capture_output=True, text=True
)
print(f"Pull stdout: {pull.stdout[-300:]}")
print(f"Pull exit code: {pull.returncode}")

# Check models
print("\nAvailable models:")
result = subprocess.run(
    [r"C:\Users\123\AppData\Local\Programs\Ollama\ollama.exe", "list"],
    capture_output=True, text=True
)
print(result.stdout)
