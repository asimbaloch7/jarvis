"""SQLite-backed persistence: recent projects and command history."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from ..logging_setup import get_logger

log = get_logger("store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    path        TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    last_opened REAL NOT NULL DEFAULT 0,
    open_count  INTEGER NOT NULL DEFAULT 0,
    created_by_jarvis INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    transcript TEXT,
    skill      TEXT,
    params     TEXT,
    status     TEXT,
    detail     TEXT
);

CREATE INDEX IF NOT EXISTS idx_history_ts ON history(ts DESC);
"""


@dataclass
class Project:
    path: str
    name: str
    last_opened: float = 0.0
    open_count: int = 0
    created_by_jarvis: bool = False

    @property
    def exists(self) -> bool:
        return Path(self.path).is_dir()


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- projects ---------------------------------------------------------

    def record_project(self, path: Path | str, created: bool = False) -> None:
        p = Path(path).expanduser().resolve()
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO projects (path, name, last_opened, open_count, created_by_jarvis)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(path) DO UPDATE SET
                last_opened = excluded.last_opened,
                open_count  = projects.open_count + 1
            """,
            (str(p), p.name, now, int(created)),
        )
        self._conn.commit()

    def add_project_if_missing(self, path: Path | str, mtime: float = 0.0) -> None:
        """Seed a project without counting it as an open."""
        p = Path(path).expanduser().resolve()
        self._conn.execute(
            """
            INSERT INTO projects (path, name, last_opened, open_count)
            VALUES (?, ?, ?, 0)
            ON CONFLICT(path) DO NOTHING
            """,
            (str(p), p.name, mtime),
        )
        self._conn.commit()

    def recent_projects(self, limit: int = 10, only_existing: bool = True) -> list[Project]:
        rows = self._conn.execute(
            "SELECT * FROM projects ORDER BY last_opened DESC, open_count DESC LIMIT ?",
            (limit * 2,),
        ).fetchall()
        projects = [
            Project(
                path=r["path"],
                name=r["name"],
                last_opened=r["last_opened"],
                open_count=r["open_count"],
                created_by_jarvis=bool(r["created_by_jarvis"]),
            )
            for r in rows
        ]
        if only_existing:
            projects = [p for p in projects if p.exists]
        return projects[:limit]

    def forget_project(self, path: Path | str) -> None:
        self._conn.execute(
            "DELETE FROM projects WHERE path = ?", (str(Path(path).expanduser().resolve()),)
        )
        self._conn.commit()

    def scan_projects(self, root: Path, max_depth: int = 2) -> int:
        """Walk `root` for git repos and seed them as known projects.

        Returns the number of newly discovered projects. Directories are ranked
        by mtime so a fresh install still gives a sensible "recent" ordering.
        """
        if not root.is_dir():
            log.warning("Projects root %s does not exist, skipping scan", root)
            return 0

        found = 0
        known = {p.path for p in self.all_projects()}

        def walk(directory: Path, depth: int) -> None:
            nonlocal found
            if depth > max_depth:
                return
            try:
                entries = list(directory.iterdir())
            except (PermissionError, OSError):
                return
            for entry in entries:
                if not entry.is_dir() or entry.name.startswith("."):
                    continue
                if (entry / ".git").exists():
                    resolved = str(entry.resolve())
                    if resolved not in known:
                        try:
                            mtime = entry.stat().st_mtime
                        except OSError:
                            mtime = 0.0
                        self.add_project_if_missing(entry, mtime)
                        known.add(resolved)
                        found += 1
                    # Don't descend into a repo looking for nested repos.
                    continue
                walk(entry, depth + 1)

        walk(root, 1)
        log.info("Project scan of %s found %d new project(s)", root, found)
        return found

    def all_projects(self) -> list[Project]:
        rows = self._conn.execute("SELECT * FROM projects").fetchall()
        return [
            Project(
                path=r["path"],
                name=r["name"],
                last_opened=r["last_opened"],
                open_count=r["open_count"],
                created_by_jarvis=bool(r["created_by_jarvis"]),
            )
            for r in rows
        ]

    # -- history ----------------------------------------------------------

    def log_interaction(
        self,
        transcript: str,
        skill: str | None,
        params: dict | None,
        status: str,
        detail: str = "",
    ) -> None:
        self._conn.execute(
            "INSERT INTO history (ts, transcript, skill, params, status, detail)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                transcript,
                skill,
                json.dumps(params or {}),
                status,
                detail[:2000],
            ),
        )
        self._conn.commit()

    def recent_history(self, limit: int = 20) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM history ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
