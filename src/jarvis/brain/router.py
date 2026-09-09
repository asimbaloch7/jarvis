"""Decides which skill handles an utterance.

Gemini first, keyword matching if Gemini is unavailable. The fallback is not
just an error path: it also means simple commands still work with no network
and no API key at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..logging_setup import get_logger
from ..skills.base import Skill
from ..skills.registry import SkillRegistry
from ..state.conversation import Conversation
from ..state.store import Store
from .gemini import BrainUnavailable, GeminiBrain
from .prompts import build_context_block

log = get_logger("brain.router")


@dataclass
class RouteDecision:
    skill: Skill | None = None
    params: dict[str, Any] = field(default_factory=dict)
    # Spoken directly to the user when there's no skill to run.
    reply: str = ""
    # "gemini" | "keyword" | "chat" | "none"
    source: str = "none"
    degraded: bool = False

    @property
    def has_skill(self) -> bool:
        return self.skill is not None


class Router:
    def __init__(self, cfg: Config, registry: SkillRegistry, store: Store):
        self.cfg = cfg
        self.registry = registry
        self.store = store
        self.brain = GeminiBrain(cfg.brain, cfg.gemini_api_key)
        self._warned_offline = False

    @property
    def gemini_enabled(self) -> bool:
        return not self.cfg.general.offline_only and self.brain.configured

    def route(self, text: str, conversation: Conversation | None = None) -> RouteDecision:
        text = (text or "").strip()
        if not text:
            return RouteDecision(reply="", source="none")

        if self.gemini_enabled:
            try:
                return self._route_with_gemini(text, conversation)
            except BrainUnavailable as exc:
                log.warning("Falling back to keyword routing: %s", exc)
                decision = self._route_with_keywords(text)
                decision.degraded = True
                if not decision.has_skill:
                    decision.reply = f"{exc.spoken} I couldn't match that to a basic command."
                elif not self._warned_offline:
                    # Mention the degradation once, then stop nagging.
                    self._warned_offline = True
                    decision.reply = exc.spoken
                return decision

        return self._route_with_keywords(text)

    def _route_with_gemini(self, text: str, conversation: Conversation | None) -> RouteDecision:
        recent = [p.name for p in self.store.recent_projects(limit=10)]
        context = build_context_block(
            recent_projects=recent,
            conversation=conversation.transcript() if conversation else "",
        )
        response = self.brain.route(text, self.registry.function_declarations(), context)

        if response.is_call:
            skill = self.registry.get(response.function_name)
            if skill is None:
                log.error("Gemini called unknown skill %r", response.function_name)
                return RouteDecision(
                    reply="I picked an action I don't actually have. That's my mistake.",
                    source="none",
                )
            return RouteDecision(skill=skill, params=response.arguments, source="gemini")

        if response.text:
            return RouteDecision(reply=response.text, source="chat")

        return RouteDecision(reply="I'm not sure what to do with that.", source="none")

    def _route_with_keywords(self, text: str) -> RouteDecision:
        match = self.registry.best_match(text)
        if match is None:
            return RouteDecision(source="none")
        skill, score = match
        log.info("Keyword routing chose %s (score %.2f)", skill.name, score)
        return RouteDecision(
            skill=skill, params=skill.extract_params(text), source="keyword"
        )
