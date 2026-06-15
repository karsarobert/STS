#!/home/karsa-robert/miniconda3/bin/python3
"""Edge-TTS Handler v3 - decodes MP3 using pydub+ffmpeg."""
import asyncio, logging, struct, io
from threading import Event
from time import perf_counter
from typing import Any, Iterator

import numpy as np
from pydub import AudioSegment
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()

VOICE = "en-US-ChristopherNeural"
OUT_RATE = 16000

class EdgeTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """Edge-TTS with pydub MP3 decoding."""

    def setup(
        self,
        should_listen: Event,
        device: str = "cpu",
        voice: str = VOICE,
        sample_rate: int = OUT_RATE,
        blocksize: int = 512,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.voice = voice
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        logger.info(f"EdgeTTSHandler ready (voice={voice}, out_rate={sample_rate}, mp3-decoded)")

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if isinstance(tts_input, EndOfResponse):
            yield AUDIO_RESPONSE_DONE
            return

        text = tts_input.text
        if not text or not text.strip():
            yield EndOfResponse()
            yield AUDIO_RESPONSE_DONE
            return

        logger.info(f"EdgeTTS: synthesizing {len(text)} chars")
        console.print(f"[green]ASSISTANT: {text.strip()}[/green]")
        start = perf_counter()

        async def _synth():
            import edge_tts as etts
            c = etts.Communicate(text, self.voice)
            buf = bytearray()
            async for chunk in c.stream():
                if chunk["type"] == "audio":
                    buf.extend(chunk["data"])
            return bytes(buf)

        mp3_data = asyncio.run(_synth())
        logger.info(f"EdgeTTS: received {len(mp3_data)} bytes MP3")

        # Decode MP3 → PCM using pydub
        seg = AudioSegment.from_mp3(io.BytesIO(mp3_data))
        # pydub returns 16-bit PCM, default sample rate
        raw_rate = seg.frame_rate
        samples = np.array(seg.get_array_of_samples(), dtype=np.int16)
        logger.info(f"EdgeTTS: decoded {len(samples)} samples @ {raw_rate}Hz ({len(samples)/raw_rate:.1f}s)")

        # Resample to pipeline rate if needed
        if raw_rate != self.sample_rate:
            from scipy.signal import resample_poly
            samples = resample_poly(samples.astype(np.float32),
                                    self.sample_rate, raw_rate).astype(np.int16)
            logger.info(f"EdgeTTS: resampled to {len(samples)} samples @ {self.sample_rate}Hz")

        dur = len(samples) / self.sample_rate
        elapsed = perf_counter() - start
        rtf = elapsed / max(dur, 0.01)
        logger.info(f"EdgeTTS: {dur:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f})")

        # Save debug WAV
        data_size = len(samples) * 2
        wav_path = "/home/karsa-robert/hermes/STS/debug_tts_out.wav"
        with open(wav_path, "wb") as f:
            f.write(b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVEfmt ")
            f.write(struct.pack("<I", 16) + struct.pack("<H", 1) + struct.pack("<H", 1))
            f.write(struct.pack("<I", self.sample_rate))
            f.write(struct.pack("<I", self.sample_rate * 2))
            f.write(struct.pack("<H", 2) + struct.pack("<H", 16) + b"data")
            f.write(struct.pack("<I", data_size) + samples.tobytes())

        # Yield chunks
        idx = 0
        while idx < len(samples):
            yield samples[idx: idx + self.blocksize]
            idx += self.blocksize

        yield EndOfResponse()
        yield AUDIO_RESPONSE_DONE

    def cleanup(self) -> None:
        pass
