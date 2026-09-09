"""How Jarvis talks and listens.

The engine only ever speaks to a Channel, so the exact same routing, skill,
and multi-turn code runs whether input comes from the microphone or from a
keyboard in `jarvis text`. That is what makes the pipeline debuggable without
audio hardware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .config import Config
from .logging_setup import get_logger

if TYPE_CHECKING:  # audio imports are deferred, see below
    from .audio.input import MicStream
    from .audio.output import Speaker
    from .stt.base import SttEngine
    from .tts.base import TtsEngine

log = get_logger("channel")


class Channel(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Deliver a message to the user."""

    @abstractmethod
    def listen(self, timeout: float | None = None) -> str | None:
        """Get the user's next utterance, or None if they said nothing."""

    def cue(self, kind: str) -> None:
        """Optional non-verbal signal: 'wake', 'done', 'error'."""


class TextChannel(Channel):
    """Keyboard in, stdout out. Used by `jarvis text` and by tests."""

    def __init__(self, echo_cues: bool = True):
        self.echo_cues = echo_cues

    def speak(self, text: str) -> None:
        if text.strip():
            print(f"\n  jarvis> {text}\n")

    def listen(self, timeout: float | None = None) -> str | None:
        try:
            return input("  you> ").strip() or None
        except (EOFError, KeyboardInterrupt):
            print()
            return None

    def cue(self, kind: str) -> None:
        if self.echo_cues:
            symbols = {"wake": "(listening)", "done": "(done)", "error": "(error)"}
            print(f"  {symbols.get(kind, kind)}")


class VoiceChannel(Channel):
    """Microphone in, TTS out.

    The audio stack is imported lazily rather than at module scope, so the
    text-mode commands (`jarvis text`, `route`, `skills`) run on a machine
    with no PortAudio and no working sound card. That matters for developing
    a skill over SSH.
    """

    def __init__(
        self,
        cfg: Config,
        mic: MicStream,
        stt: SttEngine,
        tts: TtsEngine,
        speaker: Speaker,
    ):
        self.cfg = cfg
        self.mic = mic
        self.stt = stt
        self.tts = tts
        self.speaker = speaker

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        log.info("Speaking: %s", text)
        self.tts.speak(text)

    def listen(self, timeout: float | None = None) -> str | None:
        from .audio.recorder import record_utterance

        listen_cfg = self.cfg.listen
        if timeout is not None:
            # Copy so a per-question timeout doesn't mutate global config.
            from dataclasses import replace

            listen_cfg = replace(listen_cfg, start_timeout_seconds=int(timeout))

        utterance = record_utterance(self.mic, listen_cfg)
        if not utterance.has_speech:
            return None
        text = self.stt.transcribe(utterance.audio)
        return text or None

    def cue(self, kind: str) -> None:
        if kind == "wake" and not self.cfg.tts.beep_on_wake:
            return
        self.speaker.beep(kind)
