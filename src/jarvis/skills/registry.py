"""Skill discovery.

Every module in this package is imported at startup and scanned for Skill
subclasses. Dropping a new file in `src/jarvis/skills/` is the whole
installation procedure for a new capability.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from ..logging_setup import get_logger
from .base import Skill

log = get_logger("skills.registry")


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if not skill.name:
            log.warning("Skipping %s: no name set", type(skill).__name__)
            return
        if skill.name in self._skills:
            log.warning("Duplicate skill name %r, overwriting", skill.name)
        self._skills[skill.name] = skill

    def discover(self, package: str = "jarvis.skills") -> None:
        module = importlib.import_module(package)
        for info in pkgutil.iter_modules(module.__path__):
            if info.name in ("base", "registry") or info.name.startswith("_"):
                continue
            full_name = f"{package}.{info.name}"
            try:
                submodule = importlib.import_module(full_name)
            except Exception as exc:
                log.error("Could not import skill module %s: %s", full_name, exc)
                continue
            for _, obj in inspect.getmembers(submodule, inspect.isclass):
                if (
                    issubclass(obj, Skill)
                    and obj is not Skill
                    and not inspect.isabstract(obj)
                    and obj.__module__ == full_name
                ):
                    if not obj.enabled:
                        log.info("Skill %s is disabled, skipping", obj.name)
                        continue
                    try:
                        self.register(obj())
                    except Exception as exc:
                        log.error("Could not instantiate %s: %s", obj.__name__, exc)
        log.info("Loaded %d skill(s): %s", len(self._skills), ", ".join(sorted(self._skills)))

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def all(self) -> list[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def function_declarations(self) -> list[dict]:
        return [skill.to_function_declaration() for skill in self.all()]

    def best_match(self, text: str, threshold: float = 0.4) -> tuple[Skill, float] | None:
        """Highest-scoring skill for `text`, used when Gemini is unavailable."""
        scored = [(skill, skill.match(text)) for skill in self.all()]
        scored = [(s, score) for s, score in scored if score >= threshold]
        if not scored:
            return None
        return max(scored, key=lambda pair: pair[1])

    def __len__(self) -> int:
        return len(self._skills)


def build_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.discover()
    return registry
