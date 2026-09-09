from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jarvis.channels import Channel  # noqa: E402
from jarvis.config import Config  # noqa: E402
from jarvis.state.store import Store  # noqa: E402


class ScriptedChannel(Channel):
    """A channel that answers from a fixed script.

    Lets the whole engine, including multi-turn skills, run in a test with no
    audio hardware and no network.
    """

    def __init__(self, replies: list[str] | None = None):
        self.replies = list(replies or [])
        self.spoken: list[str] = []
        self.cues: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def listen(self, timeout: float | None = None) -> str | None:
        return self.replies.pop(0) if self.replies else None

    def cue(self, kind: str) -> None:
        self.cues.append(kind)

    @property
    def said(self) -> str:
        return " | ".join(self.spoken)


@pytest.fixture
def channel():
    return ScriptedChannel()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.paths.data_dir = str(tmp_path / "data")
    cfg.paths.log_file = str(tmp_path / "state" / "jarvis.log")
    cfg.projects.root = str(tmp_path / "Workspace")
    cfg.tts.voices_dir = str(tmp_path / "voices")
    cfg.general.offline_only = True  # no network in tests
    cfg.gemini_api_key = None
    (tmp_path / "Workspace").mkdir()
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def store(config: Config) -> Store:
    s = Store(config.paths.db_path)
    yield s
    s.close()
