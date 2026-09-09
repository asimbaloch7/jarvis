"""Text-to-speech interface.

`create_tts_engine` degrades rather than crashes: if Piper isn't usable it
falls back to espeak-ng, and if that's missing too it falls back to printing,
so the rest of the pipeline stays testable on a machine with no audio output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..audio.output import Speaker
from ..config import TtsConfig
from ..logging_setup import get_logger

log = get_logger("tts")


class TtsEngine(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Say `text` out loud, blocking until finished."""

    def available(self) -> bool:
        return True

    def close(self) -> None:
        return


class NullTts(TtsEngine):
    """Prints instead of speaking. Used when no TTS backend is installed."""

    def speak(self, text: str) -> None:
        print(f"[jarvis says] {text}")
        log.info("(no TTS backend) would say: %s", text)


def create_tts_engine(cfg: TtsConfig, speaker: Speaker | None = None) -> TtsEngine:
    speaker = speaker or Speaker()
    engine = cfg.engine.lower()

    if engine == "none":
        return NullTts()

    if engine == "piper":
        from .piper_engine import PiperTts

        piper = PiperTts(cfg, speaker)
        if piper.available():
            return piper
        log.warning("Piper is not usable, falling back to espeak-ng")
        engine = "espeak"

    if engine in ("espeak", "espeak-ng"):
        from .espeak_engine import EspeakTts

        espeak = EspeakTts(cfg, speaker)
        if espeak.available():
            return espeak
        log.warning("espeak-ng is not installed either; Jarvis will print instead of speak")

    return NullTts()
