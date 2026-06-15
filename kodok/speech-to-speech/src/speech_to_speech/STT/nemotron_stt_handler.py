"""
Nemotron 3.5 ASR Speech-to-Text Handler

Connects to a local Nemotron 3.5 ASR server via WebSocket.
Supports streaming transcription with Hungarian language support.
"""
from __future__ import annotations

import asyncio
import json
import logging
from time import perf_counter
from typing import Any, Iterator, Optional

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription, VADAudio
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()

NEMOTRON_WS_URL = "ws://localhost:8002/v1/transcriptions/stream"
ASR_SAMPLE_RATE = 16000
CHUNK_MS = 560


class NemotronSTTHandler(BaseHandler[STTIn, STTOut]):
    """
    STT handler using a local Nemotron 3.5 ASR server.
    Connects via WebSocket, sends audio chunks, receives transcriptions.
    """

    def setup(
        self,
        language: str = "hu",
        url: str = NEMOTRON_WS_URL,
        chunk_ms: int = CHUNK_MS,
    ) -> None:
        self.language = language
        self.url = url
        self.chunk_ms = chunk_ms
        logger.info(f"NemotronSTTHandler ready (language={language}, url={url})")

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        is_progressive = getattr(vad_audio, "mode", None) == "progressive"
        audio_input = vad_audio.audio

        # Ensure float32 numpy array
        if not isinstance(audio_input, np.ndarray):
            audio_input = np.array(audio_input, dtype=np.float32)
        else:
            audio_input = audio_input.astype(np.float32)

        audio_duration_s = len(audio_input) / ASR_SAMPLE_RATE
        logger.debug(
            "Nemotron STT processing: mode=%s audio=%.3fs",
            "progressive" if is_progressive else "final",
            audio_duration_s,
        )

        # For progressive updates, skip (Nemotron is fast enough for final-only)
        if is_progressive:
            return

        # Run transcription via WebSocket
        try:
            text = self._transcribe(audio_input)
        except Exception as e:
            logger.error(f"Nemotron STT failed: {e}")
            text = ""

        pred_text = text.strip()
        if pred_text:
            console.print(f"[yellow]USER: {pred_text}[/yellow]")

        yield Transcription(
            text=pred_text,
            language_code=self.language,
            turn_id=getattr(vad_audio, "turn_id", None),
            turn_revision=getattr(vad_audio, "turn_revision", None),
            speech_stopped_at_s=getattr(vad_audio, "created_at_s", None),
        )

    def _transcribe(self, audio: np.ndarray) -> str:
        """Send audio to Nemotron ASR via WebSocket and return transcription."""
        import websockets

        async def _do_transcribe() -> str:
            logger.info(f"Nemotron: Connecting to {self.url} for {len(audio)} samples...")
            async with websockets.connect(self.url, max_size=None) as ws:
                # Send config
                config = {
                    "language": self.language,
                    "sample_rate": ASR_SAMPLE_RATE,
                    "chunk_ms": self.chunk_ms,
                    "use_vad": False,
                }
                await ws.send(json.dumps(config))
                logger.debug(f"Nemotron: Config sent: {config}")

                # Wait for ready
                ready = json.loads(await ws.recv())
                logger.debug(f"Nemotron: Server response: {ready}")
                if ready.get("event") != "ready":
                    raise RuntimeError(f"Nemotron ASR not ready: {ready}")

                # Send audio chunks
                chunk_samples = int(ASR_SAMPLE_RATE * self.chunk_ms / 1000)
                num_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
                logger.info(f"Nemotron: Sending {num_chunks} audio chunks ({chunk_samples} samples each)...")

                for i, start in enumerate(range(0, len(audio), chunk_samples)):
                    chunk = audio[start:start + chunk_samples]
                    # Pad last chunk if needed
                    if len(chunk) < chunk_samples:
                        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
                    await ws.send(chunk.tobytes())
                    logger.debug(f"Nemotron: Sent chunk {i+1}/{num_chunks}")

                # Signal end
                await ws.send(json.dumps({"event": "end"}))
                logger.info("Nemotron: End signal sent, waiting for transcription...")

                # Collect response
                final_text = ""
                while True:
                    try:
                        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                        event = msg.get("event")
                        logger.debug(f"Nemotron: Received event={event}")

                        if event == "partial":
                            logger.debug(f"Nemotron: Partial text: {msg.get('text', '')}")
                        elif event == "final":
                            final_text = msg.get("text", "")
                            logger.info(f"Nemotron: Final transcript: '{final_text}'")
                            break
                        elif event == "error":
                            raise RuntimeError(f"Nemotron ASR error: {msg.get('message', msg)}")
                    except asyncio.TimeoutError:
                        logger.warning("Nemotron ASR timeout - no response received")
                        break

                return final_text

        return asyncio.run(_do_transcribe())

    def cleanup(self) -> None:
        pass
