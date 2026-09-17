"""Microphone capture.

One stream stays open for the life of the daemon. The wake detector consumes
chunks from it continuously; when the wake word fires, the recorder consumes
from the same stream. Reopening the device per utterance is a reliable way to
get glitches and occasional device-busy errors out of PipeWire, so we don't.

Wake/STT always see 16 kHz mono int16. The hardware is opened at whatever
rate it actually supports (Bluetooth headsets are often 8 kHz or 48 kHz)
and we resample.
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

_BT_TOKENS = (
    "bluetooth",
    "tws",
    "airpod",
    "headset",
    "hands-free",
    "handsfree",
    "hfp",
    "a2dp",
    "pro5",
)
_FALLBACK_RATES = (16000, 48000, 44100, 32000, 8000)


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


def pick_input_device(spec: str | int | None) -> int | None:
    """Choose a capture device.

    Preference order: an explicit config value, a connected Bluetooth headset
    (AirPods / TWS), a real ALSA analog mic, then PortAudio's default.
    """
    if spec is not None:
        return resolve_device(spec, "input")

    sd = _sounddevice()
    devices = list(sd.query_devices())
    ranked: list[tuple[int, int, str]] = []
    for idx, dev in enumerate(devices):
        n_in = int(dev["max_input_channels"])
        if n_in <= 0:
            continue
        name = dev["name"]
        lowered = name.lower()
        score = 0
        if any(token in lowered for token in _BT_TOKENS):
            score += 100
            if "capture" in lowered:
                score += 20
        if "hw:" in lowered and "hdmi" not in lowered:
            score += 40
        if any(token in lowered for token in ("analog", "built-in")) and "hdmi" not in lowered:
            score += 15
        if "gnome settings" in lowered:
            score -= 30
        if n_in > 8:
            score -= 80  # PipeWire's 128-channel ALSA virtual device
        if "hdmi" in lowered:
            score -= 50
        ranked.append((score, idx, name))

    if not ranked:
        return None
    ranked.sort(key=lambda row: row[0], reverse=True)
    best_score, idx, name = ranked[0]
    if best_score <= 0:
        return None
    log.info("Using capture device %r (index %d, score %d)", name, idx, best_score)
    return idx


def _resample_int16(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.int16, copy=False)
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(src_rate), int(dst_rate))
        y = resample_poly(samples.astype(np.float32), dst_rate // g, src_rate // g)
        return np.clip(np.round(y), -32768, 32767).astype(np.int16)
    except Exception:
        n_out = max(1, int(round(samples.size * dst_rate / src_rate)))
        x_old = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        y = np.interp(x_new, x_old, samples.astype(np.float32))
        return np.clip(y, -32768, 32767).astype(np.int16)


class MicStream:
    """A continuously running 16 kHz mono int16 microphone stream."""

    def __init__(self, device: str | int | None = None, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = SAMPLE_RATE  # always 16 kHz to callers
        self.device = device
        self._native_rate = sample_rate
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._stream = None
        self._overflows = 0
        self._native_buf = np.zeros(0, dtype=np.int16)
        self.device_label = "default"

    def _enqueue(self, chunk: np.ndarray) -> None:
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk)
            except queue.Empty:
                pass

    def _ingest(self, mono: np.ndarray) -> None:
        """Accept native-rate mono int16, emit 16 kHz CHUNK_SAMPLES frames."""
        if self._native_rate == SAMPLE_RATE:
            # Split into 80 ms frames.
            if self._native_buf.size:
                mono = np.concatenate([self._native_buf, mono])
            offset = 0
            while offset + CHUNK_SAMPLES <= mono.size:
                self._enqueue(mono[offset : offset + CHUNK_SAMPLES].copy())
                offset += CHUNK_SAMPLES
            self._native_buf = mono[offset:].copy()
            return

        self._native_buf = np.concatenate([self._native_buf, mono])
        needed = max(1, int(round(CHUNK_SAMPLES * self._native_rate / SAMPLE_RATE)))
        while self._native_buf.size >= needed:
            piece = self._native_buf[:needed]
            self._native_buf = self._native_buf[needed:]
            out = _resample_int16(piece, self._native_rate, SAMPLE_RATE)
            if out.size < CHUNK_SAMPLES:
                out = np.pad(out, (0, CHUNK_SAMPLES - out.size))
            elif out.size > CHUNK_SAMPLES:
                out = out[:CHUNK_SAMPLES]
            self._enqueue(out)

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self._overflows += 1
            if self._overflows % 50 == 1:
                log.debug("Input stream status: %s", status)
        samples = np.asarray(indata)
        if samples.dtype != np.int16:
            samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        if samples.ndim == 2:
            mono = samples.mean(axis=1).astype(np.int16) if samples.shape[1] > 1 else samples[:, 0]
        else:
            mono = samples.reshape(-1)
        self._ingest(mono)

    def start(self) -> None:
        sd = _sounddevice()
        device = pick_input_device(self.device)
        last_error: Exception | None = None
        tried: list[str] = []

        max_in = 1
        native = 48000
        name = "default"
        if device is not None:
            info = sd.query_devices(device)
            name = info["name"]
            max_in = max(1, int(info["max_input_channels"]))
            native = int(info.get("default_samplerate") or 48000)

        rates = []
        for rate in (SAMPLE_RATE, native, *_FALLBACK_RATES):
            if rate not in rates:
                rates.append(rate)
        channel_options: list[int] = []
        # Mix-down from stereo first. Opening 1 channel on a 2/4-channel analog
        # device often binds to a silent monitor jack.
        if max_in >= 2:
            channel_options.append(2)
        if max_in >= 1:
            channel_options.append(1)

        for rate in rates:
            for channels in channel_options:
                label = f"{name!r} {rate} Hz {channels}ch"
                tried.append(label)
                try:
                    stream = sd.InputStream(
                        samplerate=rate,
                        blocksize=max(1, int(rate * 0.08)),
                        device=device,
                        dtype="int16",
                        channels=channels,
                        callback=self._callback,
                    )
                    stream.start()
                except Exception as exc:
                    last_error = exc
                    log.debug("Could not open %s: %s", label, exc)
                    continue
                self._stream = stream
                self._native_rate = rate
                self.device_label = f"{name} @ {rate} Hz {channels}ch"
                extra = ""
                if rate != SAMPLE_RATE:
                    extra = f", resampling to {SAMPLE_RATE} Hz"
                log.info("Microphone open (%s, %d Hz, %d ch%s)", name, rate, channels, extra)
                return

        detail = last_error or "no supported rate/channel combination"
        raise AudioUnavailable(
            f"Could not open the microphone ({detail}). "
            f"Tried: {', '.join(tried[:8])}. "
            "Run `jarvis devices` and set audio.input_device in config.yaml "
            "(for AirPods, try the name containing TWS or Bluetooth)."
        ) from last_error

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

    def drain(self) -> list[np.ndarray]:
        """Take every queued chunk without waiting. Does not touch future audio."""
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                return chunks

    def flush(self) -> None:
        """Drop buffered audio, so a recording doesn't start with stale sound."""
        self._native_buf = np.zeros(0, dtype=np.int16)
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
