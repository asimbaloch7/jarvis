"""Piper backend.

Shells out to the `piper` binary rather than importing a Python package, so
this works whether Piper came from pip, a GitHub release tarball, or a distro
package. Synthesis goes to a temp WAV which we then play ourselves, which
keeps output routing consistent with the beep cues.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ..audio.output import Speaker
from ..config import TtsConfig
from ..logging_setup import get_logger
from .base import TtsEngine

log = get_logger("tts.piper")


class PiperTts(TtsEngine):
    def __init__(self, cfg: TtsConfig, speaker: Speaker):
        self.cfg = cfg
        self.speaker = speaker
        self.binary = self._find_binary(cfg.piper_binary)
        self.model_path = self._find_voice(cfg)
        self._output_flag = "--output_file"

    @staticmethod
    def _find_binary(spec: str) -> str | None:
        candidate = Path(spec).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        return shutil.which(spec)

    @staticmethod
    def _find_voice(cfg: TtsConfig) -> Path | None:
        """Locate <voice>.onnx, checking the configured dir then common paths."""
        name = cfg.voice
        if name.endswith(".onnx"):
            direct = Path(name).expanduser()
            if direct.exists():
                return direct
            name = direct.stem

        search_dirs = [
            cfg.voices_path,
            Path("~/.local/share/piper/voices").expanduser(),
            Path("/usr/share/piper-voices"),
        ]
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            exact = directory / f"{name}.onnx"
            if exact.exists():
                return exact
            matches = sorted(directory.rglob(f"{name}.onnx"))
            if matches:
                return matches[0]
        return None

    def available(self) -> bool:
        if not self.binary:
            log.debug("piper binary not found on PATH")
            return False
        if not self.model_path:
            log.warning(
                "Piper voice %r not found in %s. Run: ./scripts/download_models.sh",
                self.cfg.voice,
                self.cfg.voices_path,
            )
            return False
        return True

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with tempfile.TemporaryDirectory(prefix="jarvis-tts-") as tmpdir:
            out = Path(tmpdir) / "speech.wav"
            if not self._synthesize(text, out):
                return
            self.speaker.play_wav(out)

    def _synthesize(self, text: str, out: Path) -> bool:
        for flag in (self._output_flag, "-f"):
            cmd = [
                self.binary,
                "--model", str(self.model_path),
                flag, str(out),
            ]
            try:
                result = subprocess.run(
                    cmd,
                    input=text.encode("utf-8"),
                    capture_output=True,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                log.error("Piper timed out synthesizing %d chars", len(text))
                return False
            except OSError as exc:
                log.error("Could not run piper: %s", exc)
                return False

            if result.returncode == 0 and out.exists() and out.stat().st_size > 44:
                self._output_flag = flag  # remember which spelling this build wants
                return True

            log.debug(
                "piper %s failed (rc=%d): %s",
                flag,
                result.returncode,
                result.stderr.decode("utf-8", "replace")[:300],
            )

        log.error("Piper synthesis failed for: %r", text[:80])
        return False
