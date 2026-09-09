"""Wake-word detector interface.

Implementations are fed 80 ms int16 chunks and return a confidence score.
Swapping openWakeWord for Porcupine means adding one file here and changing
`wake.engine` in the config.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import WakeConfig


class WakeWordDetector(ABC):
    @abstractmethod
    def score(self, chunk: np.ndarray) -> float:
        """Confidence in [0, 1] that the wake word just completed."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal buffers, e.g. after a detection is consumed."""

    def close(self) -> None:
        return


def create_wake_detector(cfg: WakeConfig) -> WakeWordDetector:
    engine = cfg.engine.lower()
    if engine == "openwakeword":
        from .openwakeword_engine import OpenWakeWordDetector

        return OpenWakeWordDetector(cfg)
    if engine == "porcupine":
        from .porcupine_engine import PorcupineDetector

        return PorcupineDetector(cfg)
    raise ValueError(f"Unknown wake engine: {cfg.engine!r}")
