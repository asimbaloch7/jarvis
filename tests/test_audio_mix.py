"""Microphone mix-down must not cancel a live channel against a silent one."""

from __future__ import annotations

import numpy as np

from jarvis.audio.input import _to_mono


def test_to_mono_picks_the_loud_channel():
    quiet = np.zeros(128, dtype=np.int16)
    loud = np.full(128, 4000, dtype=np.int16)
    stereo = np.stack([quiet, loud], axis=1)
    mono = _to_mono(stereo)
    assert mono.shape == (128,)
    assert int(np.max(np.abs(mono))) == 4000
