"""openWakeWord backend.

Runs the pretrained "hey jarvis" ONNX model. Cost is a few MB of RAM and a
small slice of one core, which is what makes always-on listening reasonable.
"""

from __future__ import annotations

import numpy as np

from ..config import WakeConfig
from ..logging_setup import get_logger
from .base import WakeWordDetector

log = get_logger("wake.openwakeword")


class OpenWakeWordDetector(WakeWordDetector):
    def __init__(self, cfg: WakeConfig):
        try:
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError as exc:
            raise RuntimeError(
                "openwakeword is not installed. Run: ./scripts/install_python_deps.sh"
            ) from exc

        model_name = cfg.model
        try:
            download_models(model_names=[model_name])
        except Exception as exc:
            # Not fatal: the models may already be cached from a previous run.
            log.debug("Model download skipped or failed (%s)", exc)

        try:
            self._model = Model(
                wakeword_models=[model_name], inference_framework="onnx"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load wake model {model_name!r}: {exc}. "
                "Try: python -c 'import openwakeword.utils as u; u.download_models()'"
            ) from exc

        self._key = self._pick_key(model_name)
        self._logged_keys = False
        log.info("Wake model %r loaded (threshold %.2f)", self._key, cfg.threshold)

    def _pick_key(self, requested: str) -> str:
        keys = list(self._model.models.keys())
        if not keys:
            raise RuntimeError("openWakeWord loaded no models")
        for key in keys:
            if key == requested or requested.startswith(key) or key in requested:
                return key
        log.warning("Model key %r not found, using %r", requested, keys[0])
        return keys[0]

    def score(self, chunk: np.ndarray) -> float:
        if chunk.dtype != np.int16:
            chunk = chunk.astype(np.int16)
        predictions = self._model.predict(chunk)
        if not self._logged_keys:
            log.info("Wake predictor keys: %s", list(predictions.keys()))
            self._logged_keys = True
        if self._key in predictions:
            return float(predictions[self._key])
        if predictions:
            return float(max(predictions.values()))
        return 0.0

    def reset(self) -> None:
        try:
            self._model.reset()
        except Exception:
            # Older versions expose prediction_buffer instead of reset().
            for buf in getattr(self._model, "prediction_buffer", {}).values():
                buf.clear()
