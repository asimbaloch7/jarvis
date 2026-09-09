"""Logging to a rotating file plus stderr."""

from __future__ import annotations

import logging
import logging.handlers
import sys

from .config import Config

_CONFIGURED = False


def setup_logging(cfg: Config, console: bool = True) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("jarvis")
    if _CONFIGURED:
        return logger

    level = getattr(logging, cfg.paths.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg.paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        cfg.paths.log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        logger.addHandler(stream)

    # These are chatty on import and during model loading.
    for noisy in ("faster_whisper", "openwakeword", "google_genai", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"jarvis.{name}")
