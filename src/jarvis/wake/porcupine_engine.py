"""Optional Porcupine backend.

Lower CPU and fewer false accepts than openWakeWord, at the cost of a free
Picovoice account. Set PICOVOICE_ACCESS_KEY and `wake.engine: porcupine`.
Porcupine ships a built-in "jarvis" keyword, so no custom .ppn is needed.
"""

from __future__ import annotations

import os

import numpy as np

from ..config import WakeConfig
from ..logging_setup import get_logger
from .base import WakeWordDetector

log = get_logger("wake.porcupine")


class PorcupineDetector(WakeWordDetector):
    def __init__(self, cfg: WakeConfig):
        try:
            import pvporcupine
        except ImportError as exc:
            raise RuntimeError(
                "pvporcupine is not installed. Run: pip install pvporcupine"
            ) from exc

        access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
        if not access_key:
            raise RuntimeError(
                "PICOVOICE_ACCESS_KEY is not set. Get a free key at console.picovoice.ai"
            )

        keyword = "jarvis" if "jarvis" in cfg.model.lower() else cfg.model
        self._porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=[keyword],
            sensitivities=[cfg.threshold],
        )
        self._frame_length = self._porcupine.frame_length
        self._buffer = np.zeros(0, dtype=np.int16)
        log.info("Porcupine loaded (keyword=%s, frame=%d)", keyword, self._frame_length)

    def score(self, chunk: np.ndarray) -> float:
        """Porcupine is binary, so this returns 1.0 or 0.0.

        It also wants its own frame length rather than our 80 ms chunks, so we
        buffer and feed it whole frames.
        """
        self._buffer = np.concatenate([self._buffer, chunk.astype(np.int16)])
        detected = False
        while len(self._buffer) >= self._frame_length:
            frame = self._buffer[: self._frame_length]
            self._buffer = self._buffer[self._frame_length :]
            if self._porcupine.process(frame) >= 0:
                detected = True
        return 1.0 if detected else 0.0

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.int16)

    def close(self) -> None:
        self._porcupine.delete()
