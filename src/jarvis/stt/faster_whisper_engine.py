"""faster-whisper backend, tuned for CPU-only transcription of short commands."""

from __future__ import annotations

import os
import time

import numpy as np

from ..config import SttConfig
from ..logging_setup import get_logger
from .base import SttEngine

log = get_logger("stt.faster_whisper")


class FasterWhisperEngine(SttEngine):
    def __init__(self, cfg: SttConfig):
        self.cfg = cfg
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install faster-whisper"
            ) from exc

        started = time.monotonic()
        # 2 physical cores; leaving it at the default oversubscribes and is
        # measurably slower on this class of CPU.
        workers = max(1, (os.cpu_count() or 2) // 2)
        self._model = WhisperModel(
            self.cfg.model,
            device="cpu",
            compute_type=self.cfg.compute_type,
            cpu_threads=workers,
        )
        log.info(
            "Whisper %s (%s) loaded in %.1fs",
            self.cfg.model,
            self.cfg.compute_type,
            time.monotonic() - started,
        )

    def unload(self) -> None:
        if self._model is not None:
            self._model = None
            log.info("Whisper unloaded")

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        self.load()
        started = time.monotonic()
        segments, _info = self._model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        log.info(
            "Transcribed %.2fs audio in %.2fs: %r",
            len(audio) / 16000,
            time.monotonic() - started,
            text,
        )
        return text
