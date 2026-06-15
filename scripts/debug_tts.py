#!/home/karsa-robert/miniconda3/bin/python3
"""Edge-TTS debug: generate audio and save to WAV."""
import asyncio, edge_tts, numpy as np, struct, sys

async def gen(text="Hello world, this is a test. The weather is nice today."):
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    
    raw = b"".join(chunks)
    samples = np.frombuffer(raw, dtype=np.int16)
    
    # Save as WAV
    rate = 24000
    n_samples = len(samples)
    data_size = n_samples * 2
    with open("/tmp/edge_tts_test.wav", "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))   # PCM
        f.write(struct.pack("<H", 1))   # mono
        f.write(struct.pack("<I", rate))
        f.write(struct.pack("<I", rate * 2))  # byte rate
        f.write(struct.pack("<H", 2))   # block align
        f.write(struct.pack("<H", 16))  # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(samples.tobytes())
    
    print(f"WAV saved: /tmp/edge_tts_test.wav")
    print(f"  Samples: {n_samples} @ {rate}Hz = {n_samples/rate:.2f}s")
    print(f"  Min: {samples.min()}, Max: {samples.max()}")
    
    # Resample to 16kHz and save that too
    from scipy.signal import resample_poly
    down = resample_poly(samples.astype(np.float32), 16000, 24000).astype(np.int16)
    rate2 = 16000
    data_size2 = len(down) * 2
    with open("/tmp/edge_tts_16k.wav", "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<I", rate2))
        f.write(struct.pack("<I", rate2 * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size2))
        f.write(down.tobytes())
    print(f"16kHz WAV: /tmp/edge_tts_16k.wav ({len(down)} samples @ {rate2}Hz = {len(down)/rate2:.2f}s)")

asyncio.run(gen())
