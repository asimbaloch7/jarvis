"""Typed configuration, merged from defaults, config.yaml, and the environment.

Precedence, lowest to highest: dataclass defaults, config/config.yaml,
JARVIS_<SECTION>_<KEY> environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _expand(path: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(path)))


@dataclass
class GeneralConfig:
    offline_only: bool = False
    idle_unload_seconds: int = 300
    conversation_turn_timeout: int = 15


@dataclass
class AudioConfig:
    input_device: str | int | None = None
    output_device: str | int | None = None
    sample_rate: int = 16000


@dataclass
class WakeConfig:
    engine: str = "openwakeword"
    model: str = "hey_jarvis_v0.1"
    threshold: float = 0.25
    cooldown_seconds: float = 1.5


@dataclass
class SttConfig:
    engine: str = "faster_whisper"
    model: str = "base.en"
    compute_type: str = "int8"
    beam_size: int = 1
    language: str = "en"


@dataclass
class ListenConfig:
    silence_ms: int = 500
    max_seconds: int = 15
    start_timeout_seconds: int = 6
    min_speech_ms: int = 250


@dataclass
class TtsConfig:
    engine: str = "piper"
    piper_binary: str = "piper"
    voice: str = "en_GB-alan-medium"
    voices_dir: str = "~/.local/share/jarvis/piper"
    beep_on_wake: bool = True

    @property
    def voices_path(self) -> Path:
        return _expand(self.voices_dir)


@dataclass
class BrainConfig:
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    timeout_seconds: int = 15
    temperature: float = 0.2


@dataclass
class ProjectsConfig:
    root: str = "~/Workspace"
    scan_depth: int = 2
    editor_command: str = "cursor"

    @property
    def root_path(self) -> Path:
        return _expand(self.root)


@dataclass
class ShellConfig:
    allowlist: list[str] = field(
        default_factory=lambda: [
            "ls", "pwd", "whoami", "date", "uptime", "df", "free",
            "uname", "hostname", "git", "systemctl", "journalctl",
            "echo", "cat", "which",
        ]
    )
    denylist_patterns: list[str] = field(
        default_factory=lambda: [
            r"\brm\s+-[a-zA-Z]*[rf]",
            r"\bdd\s+if=",
            r"\bmkfs",
            r"\bshred\b",
            r">\s*/dev/[sh]d",
            r"\bsudo\b",
            r"\bsu\b",
            r"\bchmod\s+777\s+/",
            r"curl[^|]*\|\s*(ba)?sh",
            r"wget[^|]*\|\s*(ba)?sh",
            r":\(\)\{.*\}",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\bpoweroff\b",
        ]
    )


@dataclass
class PathsConfig:
    data_dir: str = "~/.local/share/jarvis"
    log_file: str = "~/.local/state/jarvis/jarvis.log"
    log_level: str = "INFO"

    @property
    def data_path(self) -> Path:
        return _expand(self.data_dir)

    @property
    def log_path(self) -> Path:
        return _expand(self.log_file)

    @property
    def db_path(self) -> Path:
        return self.data_path / "jarvis.db"


@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    listen: ListenConfig = field(default_factory=ListenConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    projects: ProjectsConfig = field(default_factory=ProjectsConfig)
    shell: ShellConfig = field(default_factory=ShellConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    gemini_api_key: str | None = None

    def ensure_dirs(self) -> None:
        """Create the runtime directories, failing with a readable message.

        The voices directory is best-effort: not having it only means Piper
        has nowhere to look, which downgrades to espeak rather than breaking.
        """
        for label, directory in (
            ("data directory", self.paths.data_path),
            ("log directory", self.paths.log_path.parent),
        ):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot create the {label} at {directory}: {exc.strerror}. "
                    "Change paths in config.yaml, or fix the permissions."
                ) from exc

        try:
            self.tts.voices_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass


def _coerce(value: Any, target_type: Any) -> Any:
    """Best-effort coercion of a string (from env) into the field's type."""
    if not isinstance(value, str):
        return value
    origin = getattr(target_type, "__args__", None)
    types_to_try = origin if origin else (target_type,)
    for t in types_to_try:
        if t is bool:
            lowered = value.strip().lower()
            if lowered in ("1", "true", "yes", "on"):
                return True
            if lowered in ("0", "false", "no", "off"):
                return False
            continue
        if t is int:
            try:
                return int(value)
            except ValueError:
                continue
        if t is float:
            try:
                return float(value)
            except ValueError:
                continue
    return value


def _apply_dict(section: Any, values: dict[str, Any]) -> None:
    valid = {f.name: f.type for f in fields(section)}
    for key, value in values.items():
        if key in valid and value is not None:
            setattr(section, key, value)


def _apply_env(cfg: Config) -> None:
    """Map JARVIS_BRAIN_MODEL -> cfg.brain.model, and so on."""
    for section_field in fields(cfg):
        section = getattr(cfg, section_field.name)
        if not is_dataclass(section):
            continue
        # `from __future__ import annotations` makes field.type a string, so
        # resolve the real types before trying to coerce anything.
        hints = get_type_hints(type(section))
        for f in fields(section):
            env_key = f"JARVIS_{section_field.name.upper()}_{f.name.upper()}"
            if env_key in os.environ:
                setattr(
                    section, f.name, _coerce(os.environ[env_key], hints.get(f.name, str))
                )
    # A couple of shorthands that read more naturally.
    if "JARVIS_LOG_LEVEL" in os.environ:
        cfg.paths.log_level = os.environ["JARVIS_LOG_LEVEL"]


def default_config_path() -> Path:
    """config/config.yaml in the repo, falling back to ~/.config/jarvis."""
    repo_config = REPO_ROOT / "config" / "config.yaml"
    if repo_config.exists():
        return repo_config
    return _expand("~/.config/jarvis/config.yaml")


def load_config(path: Path | str | None = None) -> Config:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()  # also pick up a .env in the current working directory

    cfg = Config()
    config_path = Path(path) if path else default_config_path()
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text()) or {}
        for section_name, values in raw.items():
            section = getattr(cfg, section_name, None)
            if is_dataclass(section) and isinstance(values, dict):
                _apply_dict(section, values)

    _apply_env(cfg)
    cfg.gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
        "GOOGLE_API_KEY"
    )
    return cfg
