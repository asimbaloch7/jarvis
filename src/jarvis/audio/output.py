"""Audio playback: the wake acknowledgement tone and synthesized speech."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ..logging_setup import get_logger
from .input import AudioUnavailable, resolve_device

log = get_logger("audio.output")


def _sounddevice():
    try:
        import sounddevice as sd
    except (OSError, ImportError) as exc:
        raise AudioUnavailable(
            "PortAudio is not available for playback. Run: sudo dnf install portaudio"
        ) from exc
    return sd


def _tone(freq: float, ms: int, sample_rate: int, volume: float) -> np.ndarray:
    n = int(sample_rate * ms / 1000)
    t = np.linspace(0, ms / 1000, n, endpoint=False)
    wave_data = np.sin(2 * np.pi * freq * t)
    # 5 ms raised-cosine edges, otherwise the tone clicks.
    edge = max(1, int(sample_rate * 0.005))
    envelope = np.ones(n)
    envelope[:edge] = np.linspace(0, 1, edge)
    envelope[-edge:] = np.linspace(1, 0, edge)
    return (wave_data * envelope * volume).astype(np.float32)


class Speaker:
    """Plays raw audio and WAV files on the configured output device."""

    def __init__(self, device: str | int | None = None):
        self.device = device
        self._resolved: int | None | str = device
        self._resolve_attempted = False

    def _device(self):
        if not self._resolve_attempted:
            try:
                self._resolved = resolve_device(self.device, "output")
            except AudioUnavailable as exc:
                log.warning("%s", exc)
                self._resolved = None
            self._resolve_attempted = True
        return self._resolved

    def play(self, samples: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
        if samples.size == 0:
            return
        try:
            sd = _sounddevice()
            sd.play(samples, samplerate=sample_rate, device=self._device())
            if blocking:
                sd.wait()
        except Exception as exc:
            log.error("Playback failed: %s", exc)

    def play_wav(self, path: Path | str, blocking: bool = True) -> None:
        try:
            with wave.open(str(path), "rb") as wf:
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                width = wf.getsampwidth()
                channels = wf.getnchannels()
        except Exception as exc:
            log.error("Could not read WAV %s: %s", path, exc)
            return

        if width != 2:
            log.error("Only 16-bit WAV playback is supported (got %d-bit)", width * 8)
            return

        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels)
        self.play(samples, sample_rate, blocking=blocking)

    def beep(self, kind: str = "wake", sample_rate: int = 22050) -> None:
        """Short non-verbal cues, so Jarvis can acknowledge without talking."""
        volume = 0.18
        if kind == "wake":  # rising two-tone: "I'm listening"
            samples = np.concatenate(
                [_tone(660, 90, sample_rate, volume), _tone(990, 110, sample_rate, volume)]
            )
        elif kind == "done":  # single soft confirmation
            samples = _tone(880, 120, sample_rate, volume)
        elif kind == "error":  # falling two-tone
            samples = np.concatenate(
                [_tone(440, 110, sample_rate, volume), _tone(330, 150, sample_rate, volume)]
            )
        else:
            samples = _tone(660, 100, sample_rate, volume)
        self.play(samples, sample_rate)
