#!/home/karsa-robert/miniconda3/bin/python3
"""STS teszt: mikrofon → WebSocket → hangszóró.
Használat: felvesz 5 mp hangot, elküldi a szervernek, lejátssza a választ.
"""
import asyncio, json, struct, sys, os, time
import numpy as np
import sounddevice as sd
import websockets

WS_URL = "ws://localhost:8765/v1/realtime"
SAMPLE_RATE = 16000
RECORD_SEC = 5

async def test():
    print(f"Csatlakozás a {WS_URL} címre...")
    async with websockets.connect(WS_URL, max_size=50_000_000) as ws:
        # 1. Session created
        ev = json.loads(await ws.recv())
        print(f"  <- {ev['type']}")

        # 2. Session update (opcionális)
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"modalities": ["text", "audio"]}
        }))
        ev = json.loads(await ws.recv())
        if ev["type"] == "error":
            print(f"  - session.update nem támogatott (nem gond)")

        # 3. Felvétel
        print(f"\n🎤 Mikrofon felvétel indul ({RECORD_SEC} mp)...")
        audio = sd.rec(int(RECORD_SEC * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        for i in range(RECORD_SEC, 0, -1):
            print(f"  {i}...", end="\r")
            await asyncio.sleep(1)
        sd.wait()
        print(f"\n✅ Felvétel kész: {len(audio)} minta")

        # 4. Átalakítás PCM16 base64-re
        audio_pcm16 = (np.clip(audio.flatten(), -1, 1) * 32767).astype(np.int16)
        audio_b64 = base64_encode(audio_pcm16.tobytes())

        # 5. Küldés chunkokban
        chunk_ms = 200
        chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)
        print("📤 Audio küldése...")
        for start in range(0, len(audio_pcm16), chunk_samples):
            chunk = audio_pcm16[start:start + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
            chunk_b64 = base64_encode(chunk.tobytes())
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": chunk_b64
            }))
            await asyncio.sleep(0.01)

        # 6. Commit
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("  commit elküldve")

        # 7. Válasz fogadása
        print("\n📡 Válasz fogadása...")
        response_audio = bytearray()
        transcript = ""
        timeout = 30
        while timeout > 0:
            try:
                ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=1))
                t = ev["type"]
                if t == "response.audio_transcript.delta":
                    transcript += ev["delta"]
                elif t in ("response.audio.delta", "response.output_audio.delta"):
                    chunk = base64_decode(ev["delta"])
                    response_audio.extend(chunk)
                elif t == "response.audio.done":
                    pass
                elif t == "response.done":
                    print("  response.done")
                    break
                elif t == "error":
                    print(f"  HIBA: {ev['error']}")
                    break
            except asyncio.TimeoutError:
                timeout -= 1

        # 8. Eredmény
        print(f"\n📝 Felismert szöveg: '{transcript}'")
        print(f"🔊 Válasz audio: {len(response_audio)} bájt")

        if len(response_audio) > 0:
            # PCM16 → float32 → lejátszás
            samples = np.frombuffer(bytes(response_audio), dtype=np.int16).astype(np.float32) / 32768.0
            print(f"▶️ Lejátszás: {len(samples)/SAMPLE_RATE:.1f} mp...")
            sd.play(samples, samplerate=SAMPLE_RATE)
            sd.wait()
        else:
            print("⚠️ Nincs válasz audio")

def base64_encode(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode()

def base64_decode(s: str) -> bytes:
    import base64
    return base64.b64decode(s)

if __name__ == "__main__":
    asyncio.run(test())
