#!/home/karsa-robert/miniconda3/bin/python3
"""Simple Nemotron WebSocket test matching the actual protocol."""
import asyncio
import json
import os
import sys

# Set LD_PRELOAD before imports
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

import numpy as np
import sounddevice as sd
import websockets

WS_URL = "ws://localhost:8002/v1/transcriptions/stream"
SAMPLE_RATE = 16000
RECORD_SEC = 3
CHUNK_MS = 560

async def test():
    print(f"🔌 Connecting to {WS_URL}...")

    async with websockets.connect(WS_URL, max_size=None) as ws:
        print("✅ Connected!")

        # 1. Send config
        config = {
            "language": "hu",
            "sample_rate": SAMPLE_RATE,
            "chunk_ms": CHUNK_MS,
            "use_vad": False,
        }
        await ws.send(json.dumps(config))
        print(f"📤 Config sent: {config}")

        # 2. Wait for ready
        ready_msg = await ws.recv()
        ready = json.loads(ready_msg)
        print(f"📥 Server ready: {ready}")

        if ready.get("event") != "ready":
            print(f"❌ Server not ready: {ready}")
            return False

        # 3. Record audio
        print(f"\n🎤 Recording {RECORD_SEC} seconds... SPEAK NOW!")
        audio = sd.rec(int(RECORD_SEC * SAMPLE_RATE),
                      samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        for i in range(RECORD_SEC, 0, -1):
            print(f"  {i}...", end="\r")
            await asyncio.sleep(1)
        sd.wait()
        print(f"\n✅ Recording done: {len(audio)} samples")

        # 4. Send audio chunks (float32 bytes)
        audio_flat = audio.flatten()
        chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
        num_chunks = len(audio_flat) // chunk_samples + 1
        print(f"📤 Sending {num_chunks} audio chunks...")

        for start in range(0, len(audio_flat), chunk_samples):
            chunk = audio_flat[start:start + chunk_samples]
            # Pad last chunk if needed
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
            await ws.send(chunk.tobytes())

            # Check for partial transcription
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    data = json.loads(msg)
                    if data.get("event") == "partial":
                        print(f"  📝 Partial: {data.get('text', '')}")
            except asyncio.TimeoutError:
                pass

        print("✅ All audio sent")

        # 5. Send end signal
        await ws.send(json.dumps({"event": "end"}))
        print("📤 End signal sent")

        # 6. Wait for final transcription
        print("⏳ Waiting for final transcription...")
        for _ in range(20):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                data = json.loads(msg)
                event = data.get("event")

                if event == "partial":
                    print(f"  📝 Partial: {data.get('text', '')}")
                elif event == "final":
                    text = data.get("text", "")
                    print(f"\n{'='*60}")
                    print(f"✅ FINAL TRANSCRIPT: '{text}'")
                    print(f"{'='*60}")
                    return len(text) > 0
                elif event == "error":
                    print(f"❌ Error: {data.get('message', data)}")
                    return False
                else:
                    print(f"  📥 {event}: {data}")
            except asyncio.TimeoutError:
                print("  ⏱️ Timeout waiting for response")
                break

        print(f"\n❌ No final transcription received")
        return False

if __name__ == "__main__":
    success = asyncio.run(test())
    sys.exit(0 if success else 1)
