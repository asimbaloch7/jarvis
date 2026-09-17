"""Microphone capture.

One stream stays open for the life of the daemon. The wake detector consumes
chunks from it continuously; when the wake word fires, the recorder consumes
from the same stream.

Wake/STT always see 16 kHz mono int16. The hardware is opened at whatever
rate it actually supports (Bluetooth headsets are often 8 kHz or 48 kHz)
and we resample. Quiet laptop mics are gain-boosted so openWakeWord can
actually see the phrase.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
from collections.abc import Iterator

import numpy as np

from ..logging_setup import get_logger

log = get_logger("audio.input")

SAMPLE_RATE = 16000
# 80 ms. openWakeWord expects exactly this many samples per predict() call.
CHUNK_SAMPLES = 1280

_HEADSET_TOKENS = (
    "bluetooth",
    "bluez",
    "tws",
    "airpod",
    "headset",
    "hands-free",
    "handsfree",
    "hfp",
    "a2dp",
    "pro5",
    "jbl",
    "sony",
    "bose",
    "earbuds",
    "earphone",
    "wireless",
    "usb",
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


def capture_devices() -> list[tuple[int, dict]]:
    """PortAudio devices that can actually record."""
    return [
        (idx, dev)
        for idx, dev in enumerate(list_devices())
        if int(dev.get("max_input_channels") or 0) > 0
    ]


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


def _rank_input(idx: int, dev: dict) -> int:
    n_in = int(dev["max_input_channels"])
    if n_in <= 0:
        return -999
    name = dev["name"]
    lowered = name.lower()
    score = 0
    if any(token in lowered for token in _HEADSET_TOKENS):
        score += 100
        if "capture" in lowered or "input" in lowered or "headset" in lowered:
            score += 25
    if "hw:" in lowered and "hdmi" not in lowered:
        score += 40
    if any(token in lowered for token in ("analog", "built-in", "internal", "mic")) and "hdmi" not in lowered:
        score += 15
    if "gnome settings" in lowered:
        score -= 30
    if n_in > 8:
        score -= 80  # PipeWire's 128-channel ALSA virtual device
    if "hdmi" in lowered:
        score -= 50
    return score


def pick_input_device(spec: str | int | None) -> int | None:
    """Choose a capture device.

    Preference: explicit config, a connected headset (JBL / AirPods / BT),
    a real analog mic, then PortAudio's default.
    """
    if spec is not None:
        return resolve_device(spec, "input")

    ranked: list[tuple[int, int, str]] = []
    for idx, dev in capture_devices():
        score = _rank_input(idx, dev)
        ranked.append((score, idx, dev["name"]))

    if not ranked:
        return None
    ranked.sort(key=lambda row: row[0], reverse=True)
    best_score, idx, name = ranked[0]
    if best_score <= 0:
        return None
    log.info("Using capture device %r (index %d, score %d)", name, idx, best_score)
    return idx


def unmute_system_mic() -> None:
    """Best-effort: PipeWire/Pulse sources are often muted on laptops."""
    pactl = shutil.which("pactl")
    if not pactl:
        return
    for args in (
        [pactl, "set-source-mute", "@DEFAULT_SOURCE@", "0"],
        [pactl, "set-source-volume", "@DEFAULT_SOURCE@", "80%"],
    ):
        try:
            subprocess.run(args, capture_output=True, timeout=2, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return


def output_only_headsets() -> list[str]:
    """Bluetooth/USB gadgets that play sound but have no capture channel."""
    found: list[str] = []
    try:
        devices = list_devices()
    except AudioUnavailable:
        return found
    for dev in devices:
        name = str(dev.get("name") or "")
        lowered = name.lower()
        if not any(token in lowered for token in _HEADSET_TOKENS):
            continue
        if int(dev.get("max_input_channels") or 0) > 0:
            continue
        if int(dev.get("max_output_channels") or 0) > 0:
            found.append(name)
    return found


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


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """Collapse multi-channel audio without averaging (averaging cancels mics)."""
    if samples.ndim == 1:
        return samples
    if samples.shape[1] == 1:
        return samples[:, 0]
    # Pick the loudest channel in this block — unused/inverted channels
    # stay silent instead of wiping the real mic.
    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2, axis=0))
    return samples[:, int(np.argmax(rms))]


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
        self.device_index: int | None = None
        self._gain = 4.0  # laptop mics are usually far too quiet for openWakeWord
        self._gain_rms = 200.0

    def _enqueue(self, chunk: np.ndarray) -> None:
        if self._gain != 1.0:
            boosted = chunk.astype(np.float32) * self._gain
            chunk = np.clip(boosted, -32768, 32767).astype(np.int16)
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2))) + 1.0
        # Slow AGC: creep toward a speech-like RMS, never below 1x or above 12x.
        # Don't crank gain on a muted/silent device — that just amplifies noise.
        self._gain_rms = 0.98 * self._gain_rms + 0.02 * rms
        if rms > 80 and self._gain_rms < 1500:
            self._gain = min(12.0, self._gain * 1.002)
        elif self._gain_rms > 6000:
            self._gain = max(1.0, self._gain * 0.998)
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
        self._ingest(_to_mono(samples))

    def start(self) -> None:
        unmute_system_mic()
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

        rates: list[int] = []
        for rate in (SAMPLE_RATE, native, *_FALLBACK_RATES):
            if rate not in rates:
                rates.append(rate)
        # Prefer stereo so we can pick the loudest channel; fall back to mono.
        channel_options: list[int] = []
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
                self.device_index = device
                extra = f", resampling to {SAMPLE_RATE} Hz" if rate != SAMPLE_RATE else ""
                self.device_label = f"{name} @ {rate} Hz {channels}ch"
                log.info("Microphone open (%s, %d Hz, %d ch%s)", name, rate, channels, extra)
                return

        detail = last_error or "no supported rate/channel combination"
        raise AudioUnavailable(
            f"Could not open the microphone ({detail}). "
            f"Tried: {', '.join(tried[:8])}. "
            "Run `jarvis devices` and set audio.input_device in config.yaml."
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
        self.drain()

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
