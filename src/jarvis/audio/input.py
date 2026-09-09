"""Microphone capture.

One stream stays open for the life of the daemon. The wake detector consumes
chunks from it continuously; when the wake word fires, the recorder consumes
from the same stream. Reopening the device per utterance is a reliable way to
get glitches and occasional device-busy errors out of PipeWire, so we don't.
"""

from __future__ import annotations

import queue
from collections.abc import Iterator

import numpy as np

from ..logging_setup import get_logger

log = get_logger("audio.input")

SAMPLE_RATE = 16000
# 80 ms. openWakeWord expects exactly this many samples per predict() call.
CHUNK_SAMPLES = 1280


class AudioUnavailable(RuntimeError):
    """Raised when the mic cannot be opened, with a human-readable reason."""


def _sounddevice():
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio shared library missing
        raise AudioUnavailable(
            "PortAudio is not installed. Run: sudo dnf install portaudio"
        ) from exc
    except ImportError as exc:
        raise AudioUnavailable(
            "The sounddevice package is missing. Run: pip install sounddevice"
        ) from exc
    return sd


def list_devices() -> list[dict]:
    sd = _sounddevice()
    return list(sd.query_devices())


def resolve_device(spec: str | int | None, kind: str) -> int | None:
    """Turn a device name substring into an index; pass through ints and None."""
    if spec is None or isinstance(spec, int):
        return spec
    if isinstance(spec, str) and spec.strip().lstrip("-").isdigit():
        return int(spec.strip())
    sd = _sounddevice()
    needle = str(spec).lower()
    for idx, dev in enumerate(sd.query_devices()):
        channels = dev["max_input_channels"] if kind == "input" else dev["max_output_channels"]
        if channels > 0 and needle in dev["name"].lower():
            return idx
    raise AudioUnavailable(f"No {kind} device matching {spec!r}. Try: jarvis devices")


class MicStream:
    """A continuously running 16 kHz mono int16 microphone stream."""

    def __init__(self, device: str | int | None = None, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._stream = None
        self._overflows = 0

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self._overflows += 1
            if self._overflows % 50 == 1:
                log.debug("Input stream status: %s", status)
        try:
            self._queue.put_nowait(np.frombuffer(bytes(indata), dtype=np.int16).copy())
        except queue.Full:
            # Dropping the oldest chunk is better than blocking the audio thread.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(
                    np.frombuffer(bytes(indata), dtype=np.int16).copy()
                )
            except queue.Empty:
                pass

    def start(self) -> None:
        sd = _sounddevice()
        device = resolve_device(self.device, "input")
        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=CHUNK_SAMPLES,
                device=device,
                dtype="int16",
                channels=1,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            raise AudioUnavailable(
                f"Could not open the microphone ({exc}). "
                "Check `jarvis devices` and that something is not holding the mic."
            ) from exc
        log.info("Microphone open (device=%s, %d Hz)", device, self.sample_rate)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("Microphone closed")

    def __enter__(self) -> MicStream:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def flush(self) -> None:
        """Drop buffered audio, so a recording doesn't start with stale sound."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """One chunk of CHUNK_SAMPLES int16 samples, or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def chunks(self, timeout: float = 1.0) -> Iterator[np.ndarray]:
        while True:
            chunk = self.read(timeout=timeout)
            if chunk is not None:
                yield chunk
