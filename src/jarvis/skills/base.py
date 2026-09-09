"""The skill interface.

A skill is one class in one file. It declares what it does in plain English
(that description is what Gemini sees), a JSON-schema parameter list, and an
`execute()` method.

`execute()` can work two ways:

1. Return a `SkillResult` directly, for one-shot skills.
2. Be a generator that `yield`s `Ask(...)` objects to interrogate the user and
   receives their spoken reply as the value of the yield expression, then
   returns a `SkillResult`. The daemon drives the microphone in between, so
   skill code reads as straight-line dialogue:

       def execute(self, params, ctx):
           choice = yield Ask.choice("Existing project or new one?",
                                     ["existing", "new"])
           if choice == "new":
               name = yield Ask.free("What should I call it?")
           ...
           return SkillResult("Done.")

Follow-up answers are resolved locally when they're a yes/no or a pick-one, so
those turns cost no network round trip and work offline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Union

from ..config import Config
from ..logging_setup import get_logger
from ..state.conversation import Conversation
from ..state.store import Store


class ConversationCancelled(Exception):
    """Thrown into a skill generator when the user aborts or stops responding."""


@dataclass
class Ask:
    """A question to put to the user mid-skill."""

    prompt: str
    kind: str = "free"  # "free" | "choice" | "confirm"
    options: list[str] = field(default_factory=list)
    # Spoken back before listening again if the first answer isn't understood.
    reprompt: str | None = None
    max_attempts: int = 2

    @classmethod
    def free(cls, prompt: str, **kw: Any) -> Ask:
        return cls(prompt=prompt, kind="free", **kw)

    @classmethod
    def choice(cls, prompt: str, options: list[str], **kw: Any) -> Ask:
        return cls(prompt=prompt, kind="choice", options=list(options), **kw)

    @classmethod
    def confirm(cls, prompt: str, **kw: Any) -> Ask:
        return cls(prompt=prompt, kind="confirm", **kw)


@dataclass
class SkillResult:
    """What the user hears, plus structured detail for the log."""

    speech: str
    ok: bool = True
    data: dict[str, Any] = field(default_factory=dict)
    # Non-verbal cue to play alongside: "done", "error", or None.
    cue: str | None = None
    # Overrides the history status, which defaults to ok/failed. Use it when
    # a more precise word is useful later, e.g. "refused" or "declined".
    status: str | None = None

    @classmethod
    def error(cls, speech: str, **data: Any) -> SkillResult:
        return cls(speech=speech, ok=False, data=data, cue="error")


@dataclass
class SkillContext:
    """Everything a skill is allowed to reach for."""

    config: Config
    store: Store
    conversation: Conversation
    # Say something without ending the skill, e.g. "Working on it."
    say: Any = None

    @property
    def log(self):
        return get_logger("skill")

    def speak(self, text: str) -> None:
        if callable(self.say):
            self.say(text)


ExecuteReturn = Union[SkillResult, Generator[Ask, str, SkillResult]]


class Skill(ABC):
    """Base class for every voice-triggered capability."""

    # Function name Gemini will call. snake_case.
    name: ClassVar[str] = ""
    # Written for Gemini, not for humans: say when to use this and when not to.
    description: ClassVar[str] = ""
    # JSON Schema object. Keep it flat and use enums where possible.
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
    # Used by the offline router when Gemini is unreachable.
    keywords: ClassVar[list[str]] = []
    # If true, the daemon asks for spoken confirmation before execute() runs.
    requires_confirmation: ClassVar[bool] = False
    # Shown in `jarvis skills` and included in the Gemini system prompt.
    examples: ClassVar[list[str]] = []
    # Set false to hide a skill from routing without deleting the file.
    enabled: ClassVar[bool] = True

    @abstractmethod
    def execute(self, params: dict[str, Any], ctx: SkillContext) -> ExecuteReturn:
        """Do the thing. Return a SkillResult, or yield Asks and then return one."""

    def match(self, text: str) -> float:
        """Offline routing score in [0, 1]. Override for smarter matching."""
        lowered = text.lower()
        if not self.keywords:
            return 0.0
        hits = sum(1 for kw in self.keywords if kw.lower() in lowered)
        if hits == 0:
            return 0.0
        # Longer keyword phrases matching is stronger evidence than single words.
        best = max(
            (len(kw) for kw in self.keywords if kw.lower() in lowered), default=1
        )
        return min(1.0, 0.4 + 0.1 * hits + min(0.4, best / 50))

    def extract_params(self, text: str) -> dict[str, Any]:
        """Pull parameters out of raw text when Gemini isn't available.

        Default is empty, which means the skill runs with no arguments and
        should ask for anything it needs.
        """
        return {}

    def to_function_declaration(self) -> dict[str, Any]:
        """Gemini function-calling declaration for this skill."""
        return {
            "name": self.name,
            "description": self.description.strip(),
            "parameters": self.parameters or {"type": "object", "properties": {}},
        }

    def __repr__(self) -> str:
        return f"<Skill {self.name}>"
