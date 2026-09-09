"""Speech-to-text interface.

Engines are loaded lazily and can be unloaded, because keeping Whisper
resident while idle is the single biggest memory cost in the daemon.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import SttConfig


class SttEngine(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Safe to call repeatedly."""

    @abstractmethod
    def unload(self) -> None:
        """Release the model. The next transcribe() reloads it."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 mono 16 kHz audio in [-1, 1]."""


def create_stt_engine(cfg: SttConfig) -> SttEngine:
    engine = cfg.engine.lower()
    if engine in ("faster_whisper", "whisper"):
        from .faster_whisper_engine import FasterWhisperEngine

        return FasterWhisperEngine(cfg)
    raise ValueError(f"Unknown STT engine: {cfg.engine!r}")
