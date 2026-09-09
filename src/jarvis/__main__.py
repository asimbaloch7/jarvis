"""Entry point for `python -m jarvis` and for an editor's Run button.

Code Runner and similar tools execute this file with the system Python,
which is 3.14 on this machine and does not have Jarvis installed. If a
project venv exists, we re-exec into it before importing anything.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENV_PYTHON = _REPO_ROOT / ".venv" / "bin" / "python"


def _reexec_in_venv() -> None:
    if not _VENV_PYTHON.is_file():
        return
    try:
        if Path(sys.executable).resolve() == _VENV_PYTHON.resolve():
            return
    except OSError:
        return
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON), *sys.argv])


_reexec_in_venv()

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from jarvis.cli import main
except ModuleNotFoundError as exc:
    if exc.name and exc.name.split(".")[0] == "jarvis":
        raise
    hint = (
        f"error: missing dependency {exc.name!r} in {sys.executable}\n"
        "From the repo root run:\n"
        "    ./scripts/install_python_deps.sh\n"
        "    source .venv/bin/activate\n"
        "    jarvis doctor\n"
        "Do not execute .venv/bin/activate as a program — it has to be sourced."
    )
    sys.exit(hint)

if __name__ == "__main__":
    # A bare Run-button click has no subcommand; doctor is the useful default.
    if __package__ in (None, "") and len(sys.argv) == 1:
        sys.argv.append("doctor")
    sys.exit(main())
