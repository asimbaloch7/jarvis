"""Short-lived conversation memory for a single wake-to-completion exchange.

This exists so Gemini can see what was just said when a skill needs to
interpret a free-form follow-up ("call it invoice-parser"), and so the log
shows a whole exchange rather than disconnected fragments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" | "jarvis"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list)
    active_skill: str | None = None
    started_at: float = field(default_factory=time.time)

    def user(self, text: str) -> None:
        self.turns.append(Turn("user", text))

    def jarvis(self, text: str) -> None:
        self.turns.append(Turn("jarvis", text))

    def reset(self) -> None:
        self.turns.clear()
        self.active_skill = None
        self.started_at = time.time()

    def transcript(self, last_n: int = 6) -> str:
        speaker = {"user": "User", "jarvis": "Jarvis"}
        return "\n".join(f"{speaker[t.role]}: {t.text}" for t in self.turns[-last_n:])

    @property
    def first_user_utterance(self) -> str:
        for turn in self.turns:
            if turn.role == "user":
                return turn.text
        return ""
