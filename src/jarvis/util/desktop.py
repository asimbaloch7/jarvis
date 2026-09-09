"""Finding installed applications from freedesktop .desktop entries.

This is how `open_application` knows what "open Firefox" means without a
hardcoded table. It reads the same launchers your desktop menu shows.
"""

from __future__ import annotations

import configparser
import difflib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..logging_setup import get_logger

log = get_logger("util.desktop")

SEARCH_DIRS = [
    Path("~/.local/share/applications").expanduser(),
    Path("/usr/share/applications"),
    Path("/usr/local/share/applications"),
    Path("/var/lib/flatpak/exports/share/applications"),
    Path("~/.local/share/flatpak/exports/share/applications").expanduser(),
    Path("/var/lib/snapd/desktop/applications"),
]

# Field codes in Exec= that only make sense when launching with a file.
_FIELD_CODES = re.compile(r"%[fFuUdDnNickvm]")


@dataclass
class DesktopApp:
    name: str
    exec_command: str
    desktop_id: str
    terminal: bool = False
    keywords: str = ""

    @property
    def argv(self) -> list[str]:
        cleaned = _FIELD_CODES.sub("", self.exec_command).strip()
        # Exec values are shell-ish but rarely need a full shell; split simply.
        try:
            import shlex

            return shlex.split(cleaned)
        except ValueError:
            return cleaned.split()


@lru_cache(maxsize=1)
def list_apps() -> list[DesktopApp]:
    apps: dict[str, DesktopApp] = {}
    for directory in SEARCH_DIRS:
        if not directory.is_dir():
            continue
        for entry in sorted(directory.glob("*.desktop")):
            app = _parse(entry)
            if app and app.desktop_id not in apps:
                apps[app.desktop_id] = app
    log.debug("Found %d desktop applications", len(apps))
    return list(apps.values())


def _parse(path: Path) -> DesktopApp | None:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read(path, encoding="utf-8")
    except (configparser.Error, UnicodeDecodeError, OSError):
        return None

    section = "Desktop Entry"
    if not parser.has_section(section):
        return None
    get = lambda key, default="": parser.get(section, key, fallback=default)  # noqa: E731

    if get("Type") != "Application":
        return None
    if get("NoDisplay", "false").lower() == "true":
        return None
    if get("Hidden", "false").lower() == "true":
        return None
    name = get("Name")
    exec_command = get("Exec")
    if not name or not exec_command:
        return None

    return DesktopApp(
        name=name,
        exec_command=exec_command,
        desktop_id=path.stem,
        terminal=get("Terminal", "false").lower() == "true",
        keywords=" ".join(filter(None, [get("Keywords"), get("GenericName"), get("Comment")])),
    )


def find_app(query: str) -> DesktopApp | None:
    """Best match for a spoken app name, or None.

    Tries exact name, desktop id, a bare binary on PATH, substring, then fuzzy.
    """
    query = (query or "").strip()
    if not query:
        return None
    needle = query.lower()
    apps = list_apps()

    for app in apps:
        if app.name.lower() == needle or app.desktop_id.lower() == needle:
            return app

    # Speech-to-text loves to split or join words: "libre office", "vs code".
    squashed = needle.replace(" ", "")
    for app in apps:
        if app.name.lower().replace(" ", "") == squashed:
            return app
        if app.desktop_id.lower().split(".")[-1] == squashed:
            return app

    starts = [app for app in apps if app.name.lower().startswith(needle)]
    if len(starts) == 1:
        return starts[0]

    contains = [app for app in apps if needle in app.name.lower()]
    if len(contains) == 1:
        return contains[0]
    if contains:
        return min(contains, key=lambda a: len(a.name))

    id_matches = [app for app in apps if needle in app.desktop_id.lower()]
    if id_matches:
        return min(id_matches, key=lambda a: len(a.desktop_id))

    names = [app.name for app in apps]
    close = difflib.get_close_matches(query, names, n=1, cutoff=0.7)
    if close:
        return next(app for app in apps if app.name == close[0])

    keyword_hits = [app for app in apps if needle in app.keywords.lower()]
    if keyword_hits:
        return min(keyword_hits, key=lambda a: len(a.name))

    # Nothing in the menus, but it might still be a plain command.
    import shutil

    binary = shutil.which(query) or shutil.which(query.replace(" ", "-"))
    if binary:
        return DesktopApp(name=query, exec_command=binary, desktop_id=query)

    return None


def refresh() -> None:
    """Drop the cache after installing something new."""
    list_apps.cache_clear()


def is_wayland() -> bool:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
