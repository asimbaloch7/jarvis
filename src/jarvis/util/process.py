"""Launching and inspecting processes."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ..logging_setup import get_logger

log = get_logger("util.process")


def which(command: str) -> str | None:
    return shutil.which(command)


def is_running(pattern: str) -> bool:
    """True if any process matches `pattern` (pgrep semantics)."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        log.debug("pgrep unavailable, assuming %r is not running", pattern)
        return False


def spawn_detached(cmd: list[str], cwd: Path | str | None = None) -> subprocess.Popen | None:
    """Start a GUI app that outlives the daemon.

    `start_new_session` detaches it from our process group, so the app is not
    killed when the systemd unit restarts.
    """
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ},
        )
        log.info("Launched %s (pid %d)", " ".join(cmd), process.pid)
        return process
    except FileNotFoundError:
        log.error("Command not found: %s", cmd[0])
    except OSError as exc:
        log.error("Could not launch %s: %s", cmd, exc)
    return None


def run_capture(cmd: list[str], cwd: Path | str | None = None, timeout: int = 30):
    """Run a command and capture its output. Never raises on non-zero exit."""
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log.error("Timed out running %s", cmd)
        return subprocess.CompletedProcess(cmd, 124, "", "timed out")
    except OSError as exc:
        log.error("Could not run %s: %s", cmd, exc)
        return subprocess.CompletedProcess(cmd, 127, "", str(exc))
