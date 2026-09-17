"""Answer resolution is what makes follow-up questions work offline."""

from __future__ import annotations

import pytest

from jarvis.brain.answers import (
    is_cancel,
    resolve_choice,
    resolve_confirm,
    slugify_spoken_name,
    strip_wake_phrase,
)


@pytest.mark.parametrize(
    "said,expected",
    [
        ("yes", True),
        ("Yeah, go ahead", True),
        ("sure", True),
        ("yep please", True),
        ("um, yes", True),
        ("no", False),
        ("Nope.", False),
        ("no, don't", False),
        ("don't do that", False),
        ("what time is it", None),
        ("", None),
    ],
)
def test_resolve_confirm(said, expected):
    assert resolve_confirm(said) is expected


@pytest.mark.parametrize(
    "said,expected",
    [
        ("existing", "existing"),
        ("a new one", "new"),
        ("the existing project", "existing"),
        ("new", "new"),
    ],
)
def test_resolve_choice_existing_or_new(said, expected):
    assert resolve_choice(said, ["existing", "new"]) == expected


def test_resolve_choice_by_ordinal():
    options = ["invoice-parser", "jarvis", "portfolio-site"]
    assert resolve_choice("the second one", options) == "jarvis"
    assert resolve_choice("number three", options) == "portfolio-site"
    assert resolve_choice("the last one", options) == "portfolio-site"
    assert resolve_choice("2", options) == "jarvis"


def test_resolve_choice_by_partial_name():
    options = ["invoice-parser", "jarvis", "portfolio-site"]
    assert resolve_choice("invoice", options) == "invoice-parser"
    assert resolve_choice("open jarvis", options) == "jarvis"


def test_resolve_choice_returns_none_when_ambiguous():
    assert resolve_choice("something else entirely", ["alpha", "beta"]) is None
    assert resolve_choice("", ["alpha", "beta"]) is None


def test_resolve_choice_prefers_a_clear_winner_over_a_near_tie():
    # Two similar options and a vague answer should not silently pick one.
    assert resolve_choice("project", ["project-a", "project-b"]) is None


@pytest.mark.parametrize(
    "said,expected",
    [
        ("call it invoice parser", "invoice-parser"),
        ("My Cool App", "my-cool-app"),
        ("name it scratchpad", "scratchpad"),
        ("invoice parser dash v2", "invoice-parser-v2"),
        ("api underscore client", "api_client"),
        ("Let's call it, uh, notes!", "notes"),
    ],
)
def test_slugify_spoken_name(said, expected):
    assert slugify_spoken_name(said) == expected


def test_is_cancel():
    assert is_cancel("cancel")
    assert is_cancel("never mind")
    assert is_cancel("Forget it.")
    assert not is_cancel("open firefox")


@pytest.mark.parametrize(
    "said,expected",
    [
        ("Hey Jarvis open firefox", "open firefox"),
        ("jarvis, what's the time", "what's the time"),
        ("open firefox", "open firefox"),
        ("Hey Jarvis", ""),
    ],
)
def test_strip_wake_phrase(said, expected):
    assert strip_wake_phrase(said) == expected
