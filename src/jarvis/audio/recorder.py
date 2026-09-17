"""Turn an open mic stream into a single utterance, endpointed by silence."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import ListenConfig
from ..logging_setup import get_logger
from .input import CHUNK_SAMPLES, MicStream
from .vad import make_vad

log = get_logger("audio.recorder")

CHUNK_MS = CHUNK_SAMPLES / 16000 * 1000  # 80 ms


@dataclass
class Utterance:
    audio: np.ndarray  # float32 in [-1, 1], 16 kHz mono
    duration: float
    reason: str  # "silence" | "max_length" | "no_speech"

    @property
    def has_speech(self) -> bool:
        return self.reason != "no_speech"


def record_utterance(
    mic: MicStream,
    cfg: ListenConfig,
    preroll: list[np.ndarray] | None = None,
    flush: bool = True,
) -> Utterance:
    """Record until the speaker stops.

    `preroll` is audio captured just before this call (typically the tail of
    the wake-word buffer), prepended so we don't clip the first syllable of a
    command spoken immediately after "Hey Jarvis".

    Pass `flush=False` after a wake detection so queued command audio is kept.
    Follow-up questions should flush, or TTS playback leaks into the recording.
    """
    vad = make_vad(mic.sample_rate)
    prefix = list(preroll or [])
    collected: list[np.ndarray] = []
    silence_chunks_needed = max(1, int(cfg.silence_ms / CHUNK_MS))
    min_speech_chunks = max(1, int(cfg.min_speech_ms / CHUNK_MS))

    if flush:
        mic.flush()

    speech_chunks = 0
    trailing_silence = 0
    started = False
    start_time = time.monotonic()
    reason = "no_speech"

    while True:
        elapsed = time.monotonic() - start_time
        if not started and elapsed > cfg.start_timeout_seconds:
            log.info("No speech within %ss, giving up", cfg.start_timeout_seconds)
            break
        if elapsed > cfg.max_seconds:
            reason = "max_length"
            log.info("Hit the %ss recording cap", cfg.max_seconds)
            break

        chunk = mic.read(timeout=1.0)
        if chunk is None:
            continue

        is_speech = vad.is_speech(chunk)
        if is_speech:
            if not started:
                started = True
                log.debug("Speech started")
            speech_chunks += 1
            trailing_silence = 0
            collected.append(chunk)
        elif started:
            trailing_silence += 1
            collected.append(chunk)
            if trailing_silence >= silence_chunks_needed:
                reason = "silence"
                break
        else:
            collected.append(chunk)
            if len(collected) > 6:
                collected.pop(0)

    if speech_chunks < min_speech_chunks:
        reason = "no_speech"

    pieces = prefix + collected
    if not pieces:
        return Utterance(np.zeros(0, dtype=np.float32), 0.0, "no_speech")

    audio_int16 = np.concatenate(pieces)
    audio = audio_int16.astype(np.float32) / 32768.0
    duration = len(audio) / mic.sample_rate
    log.info("Recorded %.2fs of audio (%s)", duration, reason)
    return Utterance(audio, duration, reason)
