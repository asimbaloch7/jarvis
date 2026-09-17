"""Logging to a rotating file plus stderr, and a scannable status line on stdout."""

from __future__ import annotations

import logging
import logging.handlers
import sys

from .config import Config

_CONFIGURED = False

# Between INFO and WARNING. Used by print_status so file logs stay searchable
# without mixing into the timestamped console handler.
STATUS = 25
logging.addLevelName(STATUS, "STATUS")


class _SkipStatusFilter(logging.Filter):
    """Keep jarvis.status off the timestamped stderr stream; it has its own line."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name != "jarvis.status"


def setup_logging(cfg: Config, console: bool = True) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger("jarvis")
    if _CONFIGURED:
        return logger

    level = getattr(logging, cfg.paths.log_level.upper(), logging.INFO)
    logger.setLevel(min(level, STATUS))
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
        stream.addFilter(_SkipStatusFilter())
        logger.addHandler(stream)

    for noisy in ("faster_whisper", "openwakeword", "google_genai", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"jarvis.{name}")


def print_status(message: str, *, warn: bool = False) -> None:
    """One human-readable lifecycle line. Call only on state transitions.

    Writes immediately to stdout (flush, no extra formatting latency) and
    also records the line on jarvis.status for the log file.
    """
    prefix = "⚠️ " if warn else ""
    print(f"[jarvis] {prefix}{message}", flush=True)
    parent = logging.getLogger("jarvis")
    if not parent.handlers:
        return
    status_log = logging.getLogger("jarvis.status")
    if warn:
        status_log.warning("%s", message)
    else:
        status_log.log(STATUS, "%s", message)
