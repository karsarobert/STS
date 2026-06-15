#!/home/karsa-robert/miniconda3/bin/python3
"""Test Nemotron STT WebSocket connection."""
import asyncio
import json
import websockets
import numpy as np
import sounddevice as sd

WS_URL = "ws://localhost:8002/v1/transcriptions/stream"
SAMPLE_RATE = 16000
RECORD_SEC = 3

async def test_nemotron():
    print(f"🔌 Connecting to {WS_URL}...")

    try:
        async with websockets.connect(WS_URL) as ws:
            print("✅ Connected!")

            # Send config
            config = {
                "type": "config",
                "config": {
                    "sample_rate": SAMPLE_RATE,
                    "language": "hu"
                }
            }
            await ws.send(json.dumps(config))
            print(f"📤 Config sent: {config}")

            # Check response
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=2.0)
                print(f"📥 Server response: {response}")
            except asyncio.TimeoutError:
                print("⚠️  No config acknowledgment (may be normal)")

            # Record audio
            print(f"\n🎤 Recording {RECORD_SEC} seconds... SPEAK NOW!")
            audio = sd.rec(int(RECORD_SEC * SAMPLE_RATE),
                          samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            print(f"✅ Recording complete: {len(audio)} samples")

            # Convert to PCM16
            audio_pcm16 = (np.clip(audio.flatten(), -1, 1) * 32767).astype(np.int16)
            audio_bytes = audio_pcm16.tobytes()

            # Send in chunks (200ms each)
            chunk_size = int(SAMPLE_RATE * 0.2) * 2  # 200ms in bytes
            print(f"\n📤 Sending audio in {len(audio_bytes)//chunk_size} chunks...")

            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i+chunk_size]
                await ws.send(chunk)
                await asyncio.sleep(0.01)  # Small delay

            print("✅ All audio sent")

            # Send end signal
            await ws.send(json.dumps({"type": "finalize"}))
            print("📤 Finalize sent")

            # Wait for transcription
            print("\n⏳ Waiting for transcription...")
            timeout = 10
            transcript = ""

            while timeout > 0:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)

                    # Check if it's JSON
                    try:
                        data = json.loads(msg)
                        print(f"📥 JSON: {data}")

                        if "text" in data:
                            transcript += data["text"]
                        elif "transcript" in data:
                            transcript += data["transcript"]
                        elif "final" in data:
                            print(f"✅ Final transcription received")
                            break
                    except json.JSONDecodeError:
                        print(f"📥 Raw message: {msg[:100]}")

                except asyncio.TimeoutError:
                    timeout -= 1
                    if timeout % 2 == 0:
                        print(f"  Still waiting... ({timeout}s left)")

            print(f"\n{'='*60}")
            if transcript:
                print(f"✅ TRANSCRIPT: '{transcript}'")
                print(f"{'='*60}")
                return True
            else:
                print(f"❌ NO TRANSCRIPT RECEIVED")
                print(f"{'='*60}")
                return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import os
    os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

    success = asyncio.run(test_nemotron())
    exit(0 if success else 1)
