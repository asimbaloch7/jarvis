"""Gemini client used for intent routing via function calling.

Everything here is written to fail loudly but recoverably: any problem raises
`BrainUnavailable` with a short sentence that is safe to speak aloud, and the
router then falls back to offline keyword matching.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import BrainConfig
from ..logging_setup import get_logger
from .prompts import SYSTEM_INSTRUCTION

log = get_logger("brain.gemini")


class BrainUnavailable(Exception):
    """Gemini could not be reached or refused the request.

    `spoken` is a short phrase suitable for TTS; str() is the technical detail.
    """

    def __init__(self, detail: str, spoken: str):
        super().__init__(detail)
        self.spoken = spoken


@dataclass
class BrainResponse:
    """Either a function call, or a plain spoken reply."""

    function_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    latency: float = 0.0

    @property
    def is_call(self) -> bool:
        return self.function_name is not None


def _has_internet(host: str = "generativelanguage.googleapis.com", timeout: float = 2.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        return False


class GeminiBrain:
    def __init__(self, cfg: BrainConfig, api_key: str | None):
        self.cfg = cfg
        self.api_key = api_key
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise BrainUnavailable(
                "GEMINI_API_KEY is not set",
                "I don't have a Gemini API key, so I'm running on basic commands only.",
            )
        try:
            from google import genai
        except ImportError as exc:
            raise BrainUnavailable(
                "google-genai is not installed",
                "My language model library isn't installed.",
            ) from exc

        try:
            self._client = genai.Client(api_key=self.api_key)
        except Exception as exc:
            raise BrainUnavailable(
                f"Could not create Gemini client: {exc}",
                "I couldn't start my language model client.",
            ) from exc
        return self._client

    def _build_config(self, declarations: list[dict]):
        from google.genai import types

        tools = None
        if declarations:
            tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**decl) for decl in declarations
                    ]
                )
            ]

        kwargs: dict[str, Any] = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "temperature": self.cfg.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            # AUTO rather than ANY, so Gemini can answer conversationally when
            # no skill fits instead of forcing a bad function call.
            kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            )
        try:
            kwargs["http_options"] = types.HttpOptions(
                timeout=int(self.cfg.timeout_seconds * 1000)
            )
        except Exception:
            log.debug("This google-genai version does not accept HttpOptions.timeout")
        return types.GenerateContentConfig(**kwargs)

    def route(self, utterance: str, declarations: list[dict], context: str = "") -> BrainResponse:
        """Ask Gemini which skill to invoke for `utterance`."""
        client = self._get_client()
        config = self._build_config(declarations)

        contents = utterance if not context else f"{context}\n\nUser just said: {utterance}"

        started = time.monotonic()
        try:
            response = client.models.generate_content(
                model=self.cfg.model, contents=contents, config=config
            )
        except Exception as exc:
            raise self._classify(exc) from exc
        latency = time.monotonic() - started

        calls = getattr(response, "function_calls", None) or []
        if calls:
            call = calls[0]
            args = dict(call.args or {})
            log.info("Gemini chose %s(%s) in %.2fs", call.name, args, latency)
            return BrainResponse(function_name=call.name, arguments=args, latency=latency)

        text = (getattr(response, "text", "") or "").strip()
        log.info("Gemini replied with text in %.2fs: %r", latency, text[:120])
        return BrainResponse(text=text, latency=latency)

    def complete(self, prompt: str, system: str | None = None) -> str:
        """Plain text completion, used for interpreting free-form follow-ups."""
        from google.genai import types

        client = self._get_client()
        kwargs: dict[str, Any] = {"temperature": 0.0}
        if system:
            kwargs["system_instruction"] = system
        try:
            kwargs["http_options"] = types.HttpOptions(
                timeout=int(self.cfg.timeout_seconds * 1000)
            )
        except Exception:
            pass
        try:
            response = client.models.generate_content(
                model=self.cfg.model,
                contents=prompt,
                config=types.GenerateContentConfig(**kwargs),
            )
        except Exception as exc:
            raise self._classify(exc) from exc
        return (getattr(response, "text", "") or "").strip()

    @staticmethod
    def _classify(exc: Exception) -> BrainUnavailable:
        """Turn an SDK exception into something worth saying out loud."""
        message = str(exc)
        lowered = message.lower()

        if not _has_internet():
            return BrainUnavailable(
                f"No network: {message}",
                "I can't reach the internet, so I'm falling back to basic commands.",
            )
        if any(token in lowered for token in ("api key", "unauthenticated", "permission denied", "401", "403")):
            return BrainUnavailable(
                f"Auth failure: {message}",
                "My Gemini API key was rejected. Check the key in your env file.",
            )
        if any(token in lowered for token in ("429", "quota", "resource_exhausted", "rate limit")):
            return BrainUnavailable(
                f"Rate limited: {message}",
                "I've hit my Gemini rate limit. Try again in a moment.",
            )
        if any(token in lowered for token in ("timeout", "timed out", "deadline")):
            return BrainUnavailable(
                f"Timeout: {message}",
                "Gemini took too long to answer.",
            )
        if any(token in lowered for token in ("500", "503", "unavailable", "internal")):
            return BrainUnavailable(
                f"Upstream error: {message}",
                "Gemini is having trouble right now.",
            )
        return BrainUnavailable(
            f"Gemini call failed: {message}",
            "Something went wrong talking to Gemini.",
        )
