import logging
import threading
import time
from queue import Queue

import numpy as np
import sounddevice as sd

from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE
from speech_to_speech.pipeline.queue_types import AudioInItem, AudioOutItem

logger = logging.getLogger(__name__)


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue: Queue[AudioInItem],
        output_queue: Queue[AudioOutItem],
        should_listen: threading.Event,
        list_play_chunk_size: int = 512,
    ) -> None:
        self.list_play_chunk_size = list_play_chunk_size

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.should_listen = should_listen

    def run(self) -> None:
        # Pre-generate a static dither buffer (±1 LSB, -96 dB) to keep the
        # audio sink active without calling numpy inside the real-time callback.
        dither = np.random.randint(-1, 2, size=(self.list_play_chunk_size, 1), dtype=np.int16)

        def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time: float, status: str) -> None:
            # During shutdown, just output silence
            if self.stop_event.is_set():
                outdata[:] = 0 * outdata
                return

            if self.output_queue.empty():
                pcm = np.ascontiguousarray(indata, dtype=np.int16)
                self.input_queue.put(pcm.tobytes())
                outdata[:] = dither
            else:
                try:
                    audio_chunk = self.output_queue.get_nowait()
                    if isinstance(audio_chunk, np.ndarray):
                        logger.debug("PLAY chunk: %d samples, shape=%s, dtype=%s, min=%d max=%d",
                                     len(audio_chunk), audio_chunk.shape, audio_chunk.dtype,
                                     audio_chunk.min(), audio_chunk.max())
                        outdata[:] = audio_chunk[:, np.newaxis]
                    elif hasattr(audio_chunk, "audio"):
                        inner = audio_chunk.audio
                        if isinstance(inner, np.ndarray):
                            logger.debug("PLAY AudioOutput(np): %d samples", len(inner))
                            outdata[:] = inner[:, np.newaxis]
                        elif isinstance(inner, bytes):
                            pcm = np.frombuffer(inner, dtype=np.int16).reshape(-1, 1)
                            logger.debug("PLAY AudioOutput(bytes): %d samples", len(pcm))
                            outdata[: len(pcm)] = pcm
                            outdata[len(pcm):] = 0
                        else:
                            outdata[:] = 0 * outdata
                    elif audio_chunk == AUDIO_RESPONSE_DONE:
                        self.should_listen.set()
                        logger.debug("Response complete, listening re-enabled")
                        outdata[:] = 0 * outdata
                    else:
                        logger.debug("PLAY unknown type: %s -> silence", type(audio_chunk).__name__)
                        outdata[:] = 0 * outdata
                except Exception:
                    outdata[:] = 0 * outdata

        logger.debug("Available devices:")
        logger.debug(sd.query_devices())
        with sd.Stream(
            samplerate=16000,
            dtype="int16",
            channels=1,
            callback=callback,
            blocksize=self.list_play_chunk_size,
        ):
            logger.info("Starting local audio stream")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            print("Stopping recording")
