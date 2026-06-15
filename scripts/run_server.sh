#!/home/karsa-robert/miniconda3/bin/python3
"""STS Realtime WebSocket server."""
import os, sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

with open("/home/karsa-robert/hermes/STS/scripts/api_key.txt") as f:
    api_key = f.read().strip()

env = os.environ.copy()
env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
env["PYTHONUNBUFFERED"] = "1"

cmd = [
    "/home/karsa-robert/miniconda3/bin/speech-to-speech",
    "--mode", "realtime",
    "--stt", "parakeet-tdt",
    "--parakeet_tdt_device", "auto",
    "--tts", "edge-tts",
    "--llm_backend", "responses-api",
    "--model_name", "unsloth/gemma-4-E4B-it-GGUF",
    "--responses_api_base_url", "http://127.0.0.1:8888/v1",
    "--responses_api_api_key", api_key,
    "--ws_host", "0.0.0.0",
    "--ws_port", "8765",
    "--log_level", "INFO",
]

print(f"Starting STS WebSocket server on ws://0.0.0.0:8765/v1/realtime", flush=True)
os.execve(cmd[0], cmd, env)
