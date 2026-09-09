"""Open any installed application by name."""

from __future__ import annotations

from typing import Any

from ..util.desktop import find_app, list_apps
from ..util.process import spawn_detached
from .base import Ask, Skill, SkillContext, SkillResult


class OpenApplicationSkill(Skill):
    name = "open_application"
    description = """
    Launch an installed desktop application by name, for example Firefox,
    Spotify, the terminal, the file manager, or Slack. Use this for "open X",
    "launch X", "start X", or "bring up X" where X is an application.

    Do not use this for opening a code project; start_dev_environment handles
    that. Pass the application name exactly as the user said it, even if it is
    misspelled or spaced oddly, since the lookup is fuzzy.
    """
    parameters = {
        "type": "object",
        "properties": {
            "application": {
                "type": "string",
                "description": "Name of the application to open, as the user said it.",
            }
        },
        "required": ["application"],
    }
    keywords = ["open ", "launch ", "start ", "run app", "bring up"]
    examples = ["Open Firefox.", "Launch the terminal.", "Start Spotify."]

    def execute(self, params: dict[str, Any], ctx: SkillContext):
        query = (params.get("application") or "").strip()
        if not query:
            query = yield Ask.free("Which application?")
            query = query.strip()
        if not query:
            return SkillResult.error("I didn't catch which application you wanted.")

        app = find_app(query)
        if app is None:
            suggestion = self._closest_names(query)
            hint = f" Did you mean {suggestion}?" if suggestion else ""
            return SkillResult.error(
                f"I couldn't find an application called {query}.{hint}", query=query
            )

        argv = app.argv
        if not argv:
            return SkillResult.error(f"{app.name} has no launch command I can use.")

        if app.terminal:
            # Console apps need a terminal to live in, or they exit instantly.
            terminal_app = find_app("terminal")
            if terminal_app:
                argv = terminal_app.argv + ["-e", *argv]

        if spawn_detached(argv) is None:
            return SkillResult.error(f"I couldn't launch {app.name}.")

        return SkillResult(f"Opening {app.name}.", data={"app": app.name}, cue="done")

    @staticmethod
    def _closest_names(query: str, limit: int = 2) -> str:
        import difflib

        names = [app.name for app in list_apps()]
        close = difflib.get_close_matches(query, names, n=limit, cutoff=0.5)
        if not close:
            return ""
        return " or ".join(close)

    def extract_params(self, text: str) -> dict[str, Any]:
        lowered = text.lower()
        for verb in ("open ", "launch ", "start ", "bring up "):
            if verb in lowered:
                tail = text[lowered.index(verb) + len(verb):].strip()
                for filler in ("the ", "my ", "app ", "application "):
                    if tail.lower().startswith(filler):
                        tail = tail[len(filler):]
                tail = tail.rstrip(".!?")
                if tail:
                    return {"application": tail}
        return {}

    def match(self, text: str) -> float:
        """Only claim this utterance if it names something we can actually open."""
        base = super().match(text)
        if base == 0.0:
            return 0.0
        params = self.extract_params(text)
        target = params.get("application")
        if target and find_app(target):
            return min(1.0, base + 0.3)
        # An "open ..." phrase we can't resolve should lose to a better match.
        return base * 0.5
