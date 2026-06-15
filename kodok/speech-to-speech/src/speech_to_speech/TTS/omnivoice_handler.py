"""
OmniVoice TTS Handler

Uses k2-fsa/OmniVoice — massively multilingual zero-shot TTS (600+ languages).
0.6B model, PyTorch CUDA, Apache 2.0 license.
"""
from __future__ import annotations

import logging
import os
from threading import Event
from time import perf_counter
from typing import Any, Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()

OUT_RATE = 24000  # OmniVoice outputs at 24kHz natively


class OmniVoiceTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """OmniVoice TTS handler with voice cloning support."""

    def setup(
        self,
        should_listen: Event,
        device: str = "cuda:0",
        dtype: str = "float16",
        ref_audio: str | None = None,
        ref_text: str = "",
        instruct: str = "male, moderate pitch",
        blocksize: int = 512,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.device = device
        self.dtype_str = dtype
        self.ref_audio = ref_audio or os.environ.get("OMNIVOICE_REF_AUDIO")
        self.ref_text = ref_text or os.environ.get("OMNIVOICE_REF_TEXT", "")
        self.blocksize = blocksize
        self.instruct = instruct
        self._model = None
        logger.info(f"OmniVoiceTTSHandler ready (device={device}, instruct={instruct})")

    @property
    def model(self):
        if self._model is None:
            import torch
            from omnivoice import OmniVoice

            torch_dtype = torch.float16 if self.dtype_str == "float16" else torch.float32
            logger.info("Loading OmniVoice model (first-use download)...")
            t0 = perf_counter()
            self._model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map=self.device,
                dtype=torch_dtype,
            )
            logger.info(f"OmniVoice loaded in {perf_counter() - t0:.1f}s")
        return self._model

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        if isinstance(tts_input, EndOfResponse):
            yield AUDIO_RESPONSE_DONE
            return

        text = tts_input.text
        if not text or not text.strip():
            yield EndOfResponse()
            yield AUDIO_RESPONSE_DONE
            return

        console.print(f"[green]ASSISTANT: {text.strip()}[/green]")
        logger.info(f"OmniVoice: generating {len(text)} chars")
        start = perf_counter()

        gen_kwargs = {"text": text.strip(), "language": "hu"}

        # Voice clone mód: referencia hangfájlból (konzisztens hang)
        if self.ref_audio:
            gen_kwargs["ref_audio"] = self.ref_audio
            if self.ref_text:
                gen_kwargs["ref_text"] = self.ref_text
        # Voice design mód: instruction alapján
        elif self.instruct:
            gen_kwargs["instruct"] = self.instruct

        audio_list = self.model.generate(**gen_kwargs)
        # audio_list is list of np.ndarray at 24kHz
        samples_24k = np.concatenate(audio_list) if len(audio_list) > 1 else audio_list[0]
        samples_24k = samples_24k.astype(np.float32)
        dur_24k = len(samples_24k) / 24000

        # Resample 24kHz → 16kHz (pipeline rate)
        from scipy.signal import resample_poly
        samples_16k = resample_poly(samples_24k, 16000, 24000).astype(np.float32)

        # Normalize to int16
        peak = np.max(np.abs(samples_16k))
        if peak > 0:
            samples_16k = samples_16k / peak
        samples_int16 = (samples_16k * 32767).astype(np.int16)

        dur = len(samples_int16) / 16000
        elapsed = perf_counter() - start
        logger.info(f"OmniVoice: {dur:.1f}s audio in {elapsed:.1f}s (RTF={elapsed/max(dur,0.01):.2f})")

        # Yield audio chunks
        idx = 0
        while idx < len(samples_int16):
            yield samples_int16[idx: idx + self.blocksize]
            idx += self.blocksize

        yield EndOfResponse()
        yield AUDIO_RESPONSE_DONE

    def cleanup(self) -> None:
        self._model = None
