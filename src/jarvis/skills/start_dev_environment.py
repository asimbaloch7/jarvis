"""Start my dev environment: launch Cursor, then open or create a project.

This is the reference skill. It shows the whole multi-turn pattern: yield an
Ask, get the user's spoken reply back, branch on it, and finally return a
SkillResult. Copy this file as a starting point for anything conversational.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..brain.answers import slugify_spoken_name
from ..logging_setup import get_logger
from ..util.process import is_running, run_capture, spawn_detached, which
from .base import Ask, Skill, SkillContext, SkillResult

log = get_logger("skill.start_dev_environment")

# Directories that are never useful as a project name.
_JUNK_NAMES = {"", ".", "..", "~", "home"}


class StartDevEnvironmentSkill(Skill):
    name = "start_dev_environment"
    description = """
    Start the user's development environment. Launches the Cursor editor and
    then opens a project in it, either an existing project or a brand new one.
    Use this for phrases like "start my dev environment", "set up my
    workspace", "let's start coding", "open my editor", "make a new project",
    or "open the <name> project".

    If the user did not say whether they want an existing or a new project,
    omit the "mode" parameter and the skill will ask them out loud. Likewise,
    omit "project" if they did not name one. Do not guess a project name.
    """
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["existing", "new"],
                "description": (
                    "Whether to open an existing project or create a new one. "
                    "Omit entirely if the user did not make this clear."
                ),
            },
            "project": {
                "type": "string",
                "description": (
                    "Name of the project to open, or the name for a new project, "
                    "exactly as the user said it. Omit if not mentioned."
                ),
            },
        },
    }
    keywords = [
        "dev environment", "development environment", "start coding",
        "start my dev", "open cursor", "launch cursor", "new project",
        "open project", "my workspace", "start working",
    ]
    examples = [
        "Start my dev environment.",
        "Open the invoice parser project.",
        "Make me a new project called scratchpad.",
    ]

    # -- entry point ------------------------------------------------------

    def execute(self, params: dict[str, Any], ctx: SkillContext):
        editor = ctx.config.projects.editor_command
        if which(editor) is None:
            return SkillResult.error(
                f"I can't find {editor} on your path, so I can't start your dev environment.",
                editor=editor,
            )

        launched_editor = self._ensure_editor_running(editor)

        mode = (params.get("mode") or "").strip().lower()
        requested = (params.get("project") or "").strip()

        if mode not in ("existing", "new"):
            # If they named a project that already exists, we don't need to ask.
            if requested and self._find_project(requested, ctx):
                mode = "existing"
            else:
                answer = yield Ask.choice(
                    "Do you want to open an existing project, or start a new one?",
                    ["existing", "new"],
                    reprompt="Sorry. Existing project, or new project?",
                )
                mode = answer

        if mode == "new":
            result = yield from self._create_new(requested, ctx, editor)
        else:
            result = yield from self._open_existing(requested, ctx, editor)

        if launched_editor and result.ok:
            result.speech = f"Cursor is up. {result.speech}"
        return result

    # -- branches ---------------------------------------------------------

    def _open_existing(self, requested: str, ctx: SkillContext, editor: str):
        projects = ctx.store.recent_projects(limit=8)

        if requested:
            match = self._find_project(requested, ctx)
            if match:
                return self._open(Path(match), ctx, editor)

        if not projects:
            ctx.log.info("No known projects; offering to create one instead")
            wants_new = yield Ask.confirm(
                "I don't have any projects on record yet. Shall I create a new one?"
            )
            if not wants_new:
                return SkillResult("Alright, nothing opened.", cue="done")
            return (yield from self._create_new("", ctx, editor))

        names = [p.name for p in projects]
        spoken = self._spoken_list(names)
        prompt = (
            f"I couldn't find {requested}. " if requested else ""
        ) + f"Which project? I know {spoken}."

        choice = yield Ask.choice(
            prompt,
            names,
            reprompt=f"Say the name or the number. {spoken}.",
        )
        selected = next(p for p in projects if p.name == choice)
        return self._open(Path(selected.path), ctx, editor)

    def _create_new(self, requested: str, ctx: SkillContext, editor: str):
        root = ctx.config.projects.root_path
        name = slugify_spoken_name(requested) if requested else ""

        attempts = 0
        while not name or name in _JUNK_NAMES:
            if attempts >= 2:
                return SkillResult.error("I didn't catch a usable project name. Stopping there.")
            spoken = yield Ask.free(
                "What should I call it?"
                if attempts == 0
                else "I didn't get that. What name should the project have?"
            )
            name = slugify_spoken_name(spoken)
            attempts += 1

        target = root / name
        if target.exists():
            open_it = yield Ask.confirm(
                f"{self._speakable(name)} already exists. Do you want me to open it instead?"
            )
            if open_it:
                return self._open(target, ctx, editor)
            return SkillResult("Alright, I'll leave it alone.", cue="done")

        try:
            target.mkdir(parents=True)
        except OSError as exc:
            ctx.log.error("Could not create %s: %s", target, exc)
            return SkillResult.error(
                f"I couldn't create that folder. {exc.strerror or 'Check the path and permissions.'}"
            )

        self._scaffold(target, name)
        git_note = self._git_init(target)

        ctx.store.record_project(target, created=True)
        result = self._open(target, ctx, editor, created=True)
        if result.ok:
            result.speech = (
                f"Created {self._speakable(name)} in {root.name}{git_note}, and opened it in Cursor."
            )
        return result

    # -- helpers ----------------------------------------------------------

    def _ensure_editor_running(self, editor: str) -> bool:
        """Start the editor if it isn't up. Returns True if we started it.

        Opening a folder also starts the editor, so this only matters for the
        spoken confirmation and for the case where the user wants the editor
        up while they decide what to open.
        """
        if is_running(f"/{editor}"):
            log.debug("%s already running", editor)
            return False
        spawn_detached([editor])
        return True

    def _find_project(self, requested: str, ctx: SkillContext) -> str | None:
        """Resolve a spoken project name against known projects, then the disk."""
        from ..brain.answers import resolve_choice

        projects = ctx.store.recent_projects(limit=50)
        if projects:
            match = resolve_choice(requested, [p.name for p in projects])
            if match:
                return next(p.path for p in projects if p.name == match)

        # Not in the database, but it might be a folder we've never opened.
        root = ctx.config.projects.root_path
        slug = slugify_spoken_name(requested)
        for candidate in {requested, slug, requested.replace(" ", "_")}:
            path = root / candidate
            if path.is_dir():
                return str(path)
        return None

    def _open(
        self, path: Path, ctx: SkillContext, editor: str, created: bool = False
    ) -> SkillResult:
        if not path.is_dir():
            return SkillResult.error(
                f"{self._speakable(path.name)} isn't there any more. I've removed it from my list.",
                path=str(path),
            )
        process = spawn_detached([editor, str(path)])
        if process is None:
            return SkillResult.error(f"I couldn't launch {editor}.", path=str(path))

        ctx.store.record_project(path, created=created)
        return SkillResult(
            f"Opened {self._speakable(path.name)} in Cursor.",
            data={"path": str(path), "created": created},
            cue="done",
        )

    @staticmethod
    def _scaffold(target: Path, name: str) -> None:
        """A README and a .gitignore, so the first commit isn't empty."""
        try:
            (target / "README.md").write_text(f"# {name}\n")
            (target / ".gitignore").write_text(
                "__pycache__/\n*.py[cod]\n.venv/\nnode_modules/\n.env\ndist/\nbuild/\n"
            )
        except OSError as exc:
            log.warning("Could not scaffold %s: %s", target, exc)

    @staticmethod
    def _git_init(target: Path) -> str:
        """Initialise a repo and make the first commit. Returns a spoken suffix."""
        if which("git") is None:
            return ""
        result = run_capture(["git", "init", "-q"], cwd=target)
        if result.returncode != 0:
            log.warning("git init failed in %s: %s", target, result.stderr.strip())
            return ""
        run_capture(["git", "add", "-A"], cwd=target)
        commit = run_capture(
            ["git", "commit", "-q", "-m", "Initial commit"], cwd=target
        )
        if commit.returncode != 0:
            # Almost always missing user.name/user.email on a fresh machine.
            log.info("Initial commit skipped: %s", commit.stderr.strip()[:200])
            return ", with git initialised"
        return ", with git initialised"

    @staticmethod
    def _speakable(name: str) -> str:
        """Hyphens and underscores read badly through TTS."""
        return name.replace("-", " ").replace("_", " ")

    @staticmethod
    def _spoken_list(names: list[str], limit: int = 5) -> str:
        shown = [n.replace("-", " ").replace("_", " ") for n in names[:limit]]
        if len(shown) == 1:
            return shown[0]
        listed = ", ".join(shown[:-1]) + f", and {shown[-1]}"
        if len(names) > limit:
            listed += f", plus {len(names) - limit} more"
        return listed

    def extract_params(self, text: str) -> dict[str, Any]:
        """Offline parameter extraction, used when Gemini is unreachable."""
        lowered = text.lower()
        params: dict[str, Any] = {}
        if any(word in lowered for word in ("new project", "create a project", "start a new")):
            params["mode"] = "new"
        elif any(word in lowered for word in ("existing", "open the", "open my")):
            params["mode"] = "existing"

        for marker in ("called ", "named ", "project "):
            if marker in lowered:
                tail = text[lowered.index(marker) + len(marker):].strip()
                if tail and len(tail) < 60:
                    params["project"] = tail
                    break
        return params
