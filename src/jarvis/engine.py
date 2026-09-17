"""The part that turns a transcript into an action.

Owns the skill registry, the router, and the dialogue driver that pumps a
skill's generator: yield an Ask, put the question to the user through the
channel, resolve their answer, send it back in. Input-agnostic by design.
"""

from __future__ import annotations

import inspect
import traceback
from dataclasses import dataclass

from .brain.answers import is_cancel, resolve_choice, resolve_confirm, resolve_free
from .brain.gemini import BrainUnavailable
from .brain.router import Router
from .channels import Channel
from .config import Config
from .logging_setup import get_logger, print_status
from .skills.base import (
    Ask,
    ConversationCancelled,
    Skill,
    SkillContext,
    SkillResult,
)
from .skills.registry import SkillRegistry, build_registry
from .state.conversation import Conversation
from .state.store import Store

log = get_logger("engine")

# Distinguishes "generator not started" from a legitimate falsy answer such as
# a "no" to a confirm question.
_START = object()


@dataclass
class Outcome:
    """What happened with one utterance, for the caller and the log."""

    spoken: str = ""
    skill: str | None = None
    ok: bool = True
    handled: bool = False


class Engine:
    def __init__(
        self,
        cfg: Config,
        channel: Channel,
        store: Store | None = None,
        registry: SkillRegistry | None = None,
    ):
        self.cfg = cfg
        self.channel = channel
        self.store = store or Store(cfg.paths.db_path)
        self.registry = registry or build_registry()
        self.router = Router(cfg, self.registry, self.store)
        self.conversation = Conversation()

    # -- public -----------------------------------------------------------

    def handle(self, text: str) -> Outcome:
        """Route one utterance and run whatever it maps to."""
        text = (text or "").strip()
        if not text:
            return Outcome()

        self.conversation.reset()
        self.conversation.user(text)
        log.info("Utterance: %r", text)

        if not self.channel.mark_processing_announced():
            print_status("⚙️  Processing your request...")

        if is_cancel(text):
            self._say("Never mind, then.")
            return Outcome(spoken="Never mind, then.", handled=True)

        decision = self.router.route(text, self.conversation)

        # The router may have something to say before or instead of acting
        # (a degradation notice, or a plain conversational answer).
        if decision.reply and not decision.has_skill:
            self._say(decision.reply)
            self.store.log_interaction(text, None, {}, decision.source, decision.reply)
            return Outcome(spoken=decision.reply, handled=decision.source == "chat")

        if decision.reply:
            self._say(decision.reply)

        if not decision.has_skill:
            message = "I don't know how to do that yet."
            if not self.router.gemini_enabled:
                message += " I'm only matching basic keywords at the moment."
            self._say(message)
            self.store.log_interaction(text, None, {}, "unmatched", "")
            return Outcome(spoken=message)

        return self._invoke(decision.skill, decision.params, text, decision.source)

    def close(self) -> None:
        self.store.close()

    # -- skill invocation -------------------------------------------------

    def _invoke(self, skill: Skill, params: dict, transcript: str, source: str) -> Outcome:
        log.info("Invoking %s(%s) via %s", skill.name, params, source)
        self.conversation.active_skill = skill.name
        ctx = SkillContext(
            config=self.cfg,
            store=self.store,
            conversation=self.conversation,
            say=self._say,
        )

        if skill.requires_confirmation:
            approved = self._ask(
                Ask.confirm(f"Do you want me to {skill.name.replace('_', ' ')}?")
            )
            if approved is None or not approved:
                self._say("Cancelled.")
                return Outcome(spoken="Cancelled.", skill=skill.name)

        try:
            result = self._drive(skill, params, ctx)
        except ConversationCancelled:
            message = "Cancelled."
            self._say(message)
            self.store.log_interaction(transcript, skill.name, params, "cancelled", "")
            return Outcome(spoken=message, skill=skill.name, handled=True)
        except Exception as exc:
            log.error("Skill %s crashed: %s\n%s", skill.name, exc, traceback.format_exc())
            print_status(f"Skill {skill.name} crashed: {exc}", warn=True)
            message = f"Something went wrong running {skill.name.replace('_', ' ')}."
            self.channel.cue("error")
            self._say(message)
            self.store.log_interaction(
                transcript, skill.name, params, "crashed", f"{type(exc).__name__}: {exc}"
            )
            return Outcome(spoken=message, skill=skill.name, ok=False, handled=True)

        if result is None:
            result = SkillResult("Done.", cue="done")

        if result.cue:
            self.channel.cue(result.cue)
        self._say(result.speech)
        self.store.log_interaction(
            transcript,
            skill.name,
            params,
            result.status or ("ok" if result.ok else "failed"),
            result.speech,
        )
        return Outcome(
            spoken=result.speech, skill=skill.name, ok=result.ok, handled=True
        )

    def _drive(self, skill: Skill, params: dict, ctx: SkillContext) -> SkillResult | None:
        """Run a skill, pumping its Ask/reply generator if it uses one."""
        outcome = skill.execute(params, ctx)

        if not inspect.isgenerator(outcome):
            return outcome

        generator = outcome
        to_send: object = _START
        try:
            while True:
                ask = next(generator) if to_send is _START else generator.send(to_send)
                if not isinstance(ask, Ask):
                    raise TypeError(
                        f"{skill.name} yielded {type(ask).__name__}, expected Ask"
                    )
                answer = self._ask(ask)
                if answer is None:
                    generator.throw(ConversationCancelled())
                    return None
                to_send = answer
        except StopIteration as stop:
            return stop.value

    # -- dialogue ---------------------------------------------------------

    def _ask(self, ask: Ask):
        """Put a question to the user and resolve their answer.

        Returns the resolved answer (str for free/choice, bool for confirm),
        or None if they cancelled, went silent, or we gave up understanding.
        """
        prompt = ask.prompt
        for attempt in range(max(1, ask.max_attempts)):
            self._say(prompt)
            reply = self.channel.listen(timeout=self.cfg.general.conversation_turn_timeout)

            if reply is None or not reply.strip():
                log.info("No reply to %r (attempt %d)", ask.prompt[:50], attempt + 1)
                prompt = ask.reprompt or f"I didn't catch that. {ask.prompt}"
                continue

            log.info("Reply: %r", reply)
            self.conversation.user(reply)

            if is_cancel(reply):
                log.info("User cancelled during follow-up")
                return None

            resolved = self._resolve(ask, reply)
            if resolved is not None:
                log.info("Resolved %s answer to %r", ask.kind, resolved)
                return resolved

            prompt = ask.reprompt or self._default_reprompt(ask)

        log.info("Gave up on question after %d attempts", ask.max_attempts)
        return None

    def _resolve(self, ask: Ask, reply: str):
        if ask.kind == "confirm":
            return resolve_confirm(reply)
        if ask.kind == "choice":
            local = resolve_choice(reply, ask.options)
            if local is not None:
                return local
            return self._resolve_choice_with_gemini(ask, reply)
        cleaned = resolve_free(reply)
        return cleaned or None

    def _resolve_choice_with_gemini(self, ask: Ask, reply: str) -> str | None:
        """Escalate an ambiguous pick-one answer, if Gemini is available.

        Local matching handles the normal cases; this catches things like
        "the one I was working on yesterday" or a badly transcribed name.
        """
        if not self.router.gemini_enabled:
            return None
        options = "\n".join(f"- {opt}" for opt in ask.options)
        prompt = (
            f"The assistant asked: {ask.prompt}\n"
            f"The available options are:\n{options}\n\n"
            f"The user replied: {reply}\n\n"
            "Reply with exactly one option from the list, copied verbatim, and "
            "nothing else. If none of them plausibly matches, reply with the "
            "single word NONE."
        )
        try:
            answer = self.router.brain.complete(prompt).strip()
        except BrainUnavailable as exc:
            log.debug("Could not escalate choice to Gemini: %s", exc)
            return None

        if answer.upper().startswith("NONE"):
            return None
        # Trust but verify: it must be one of the options we offered.
        for option in ask.options:
            if option.lower() == answer.lower():
                return option
        return resolve_choice(answer, ask.options)

    @staticmethod
    def _default_reprompt(ask: Ask) -> str:
        if ask.kind == "confirm":
            return "Sorry, was that a yes or a no?"
        if ask.kind == "choice" and ask.options:
            listed = ", ".join(ask.options[:5])
            return f"I didn't follow. Please pick one of: {listed}."
        return f"Sorry, say that again. {ask.prompt}"

    def _say(self, text: str) -> None:
        if not text:
            return
        self.conversation.jarvis(text)
        self.channel.speak(text)
