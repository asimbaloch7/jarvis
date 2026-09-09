"""Voice activity detection.

Uses webrtcvad when it is installed, because it is more robust against
steady background noise. Falls back to an adaptive energy gate, which needs no
extra dependency and works fine in a quiet room. Both expose the same
`is_speech(chunk)` interface.
"""

from __future__ import annotations

import numpy as np

from ..logging_setup import get_logger

log = get_logger("audio.vad")


class EnergyVad:
    """RMS gate with a noise floor that tracks the room while you're silent.

    The floor only adapts downward-ish during non-speech, so a long utterance
    can't drag the threshold up and cut itself off.
    """

    def __init__(
        self,
        speech_ratio: float = 3.0,
        absolute_floor: float = 180.0,
        adapt_rate: float = 0.05,
    ):
        self.speech_ratio = speech_ratio
        self.absolute_floor = absolute_floor
        self.adapt_rate = adapt_rate
        self.noise_floor = absolute_floor
        self._calibrated = False

    def calibrate(self, chunks: list[np.ndarray]) -> None:
        if not chunks:
            return
        levels = [self._rms(c) for c in chunks]
        self.noise_floor = max(float(np.median(levels)), 40.0)
        self._calibrated = True
        log.debug("VAD calibrated: noise floor %.1f", self.noise_floor)

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        samples = chunk.astype(np.float32)
        return float(np.sqrt(np.mean(samples * samples)))

    @property
    def threshold(self) -> float:
        return max(self.noise_floor * self.speech_ratio, self.absolute_floor)

    def is_speech(self, chunk: np.ndarray) -> bool:
        rms = self._rms(chunk)
        speaking = rms > self.threshold
        if not speaking:
            self.noise_floor = (
                1 - self.adapt_rate
            ) * self.noise_floor + self.adapt_rate * rms
            self.noise_floor = max(self.noise_floor, 20.0)
        return speaking

    def reset(self) -> None:
        self.noise_floor = self.absolute_floor
        self._calibrated = False


class WebRtcVad:
    """Thin wrapper over webrtcvad, chunked into the 30 ms frames it requires."""

    FRAME_MS = 30

    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000):
        import webrtcvad  # imported lazily; optional dependency

        self._vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self._frame_samples = int(sample_rate * self.FRAME_MS / 1000)

    def calibrate(self, chunks: list[np.ndarray]) -> None:  # noqa: D102 - parity with EnergyVad
        return

    def reset(self) -> None:  # noqa: D102 - parity with EnergyVad
        return

    def is_speech(self, chunk: np.ndarray) -> bool:
        """True if any full frame inside the chunk contains speech."""
        data = chunk.astype(np.int16).tobytes()
        frame_bytes = self._frame_samples * 2
        for offset in range(0, len(data) - frame_bytes + 1, frame_bytes):
            if self._vad.is_speech(data[offset : offset + frame_bytes], self.sample_rate):
                return True
        return False


def make_vad(sample_rate: int = 16000):
    """Prefer webrtcvad, fall back to the energy gate."""
    try:
        vad = WebRtcVad(sample_rate=sample_rate)
        log.debug("Using webrtcvad")
        return vad
    except Exception:
        log.debug("webrtcvad unavailable, using energy VAD")
        return EnergyVad()
