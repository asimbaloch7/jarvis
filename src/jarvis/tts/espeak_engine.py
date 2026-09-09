"""espeak-ng fallback: robotic, but instant and always available via dnf."""

from __future__ import annotations

import shutil
import subprocess

from ..audio.output import Speaker
from ..config import TtsConfig
from ..logging_setup import get_logger
from .base import TtsEngine

log = get_logger("tts.espeak")


class EspeakTts(TtsEngine):
    def __init__(self, cfg: TtsConfig, speaker: Speaker):
        self.cfg = cfg
        self.speaker = speaker
        self.binary = shutil.which("espeak-ng") or shutil.which("espeak")
        # Approximate the configured Piper voice's accent where we can.
        self.voice = "en-gb" if "en_GB" in cfg.voice else "en-us"

    def available(self) -> bool:
        return self.binary is not None

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text or not self.binary:
            return
        try:
            subprocess.run(
                [self.binary, "-v", self.voice, "-s", "165", text],
                capture_output=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.error("espeak failed: %s", exc)
