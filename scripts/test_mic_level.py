#!/home/karsa-robert/miniconda3/bin/python3
"""Mikrofon szint tesztelő - real-time audio level monitor."""
import os
import sys

# libstdc++ fix - ugyanaz, mint a run_local.py-ban
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK_DURATION = 100  # ms

def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)

    # RMS érték számítás
    volume_norm = np.linalg.norm(indata) * 10
    # Peak érték
    peak = np.abs(indata).max()

    # Vizuális bar
    bar_length = int(volume_norm * 2)
    bar = '█' * min(bar_length, 50)

    print(f"\rVolume: {volume_norm:05.2f} |{bar:<50}| Peak: {peak:.3f}", end='', flush=True)

print("Elérhető eszközök:")
print(sd.query_devices())
print("\n" + "="*60)
print("Default input device:", sd.default.device[0])
print("="*60)

# Válassz eszközt
device_id = None  # None = default
# Ha át szeretnéd állítani, írd be az eszköz számát:
# device_id = 1  # pl. USB Audio Device

if device_id is not None:
    sd.default.device[0] = device_id
    print(f"Mikrofon beállítva: {device_id}")

print("\n🎤 Beszélj a mikrofonba! (Ctrl+C kilépés)\n")

try:
    with sd.InputStream(
        device=device_id,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=int(SAMPLE_RATE * BLOCK_DURATION / 1000),
        callback=audio_callback
    ):
        print("Recording...")
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\n\nKész!")
except Exception as e:
    print(f"\n\nHiba: {e}", file=sys.stderr)
    sys.exit(1)
