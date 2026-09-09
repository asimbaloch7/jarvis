"""End-to-end dialogue tests: the engine driving real skills, no hardware."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.engine import Engine
from jarvis.skills.base import Ask, Skill, SkillContext, SkillResult
from jarvis.skills.registry import SkillRegistry


class GreetSkill(Skill):
    """One-shot skill: returns without asking anything."""

    name = "greet"
    description = "Say hello."
    keywords = ["hello", "hi there"]

    def execute(self, params, ctx):
        return SkillResult("Hello.")


class PizzaSkill(Skill):
    """Multi-turn skill exercising all three Ask kinds."""

    name = "order_pizza"
    description = "Order a pizza."
    keywords = ["pizza"]

    def execute(self, params, ctx):
        size = yield Ask.choice("What size?", ["small", "large"])
        name = yield Ask.free("Whose name?")
        confirmed = yield Ask.confirm("Place the order?")
        if not confirmed:
            return SkillResult("Order cancelled.")
        return SkillResult(f"Ordered a {size} pizza for {name}.")


class ExplodingSkill(Skill):
    name = "explode"
    description = "Always fails."
    keywords = ["explode"]

    def execute(self, params, ctx):
        raise RuntimeError("boom")


def make_engine(config, channel, skills):
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return Engine(config, channel, registry=registry)


def test_single_turn_skill(config, channel):
    engine = make_engine(config, channel, [GreetSkill()])
    outcome = engine.handle("hello")
    assert outcome.skill == "greet"
    assert outcome.ok
    assert "Hello." in channel.said
    engine.close()


def test_multi_turn_collects_every_answer(config, channel):
    channel.replies = ["large", "Asim", "yes"]
    engine = make_engine(config, channel, [PizzaSkill()])
    outcome = engine.handle("order a pizza")
    assert outcome.ok
    assert "Ordered a large pizza for asim." in channel.said
    engine.close()


def test_multi_turn_respects_a_no(config, channel):
    channel.replies = ["small", "Asim", "no thanks"]
    engine = make_engine(config, channel, [PizzaSkill()])
    engine.handle("order a pizza")
    assert "Order cancelled." in channel.said
    engine.close()


def test_reprompts_once_then_gives_up(config, channel):
    # Neither reply resolves to a size, so the skill should be abandoned.
    channel.replies = ["banana", "kumquat"]
    engine = make_engine(config, channel, [PizzaSkill()])
    outcome = engine.handle("order a pizza")
    assert "Cancelled." in channel.said
    assert outcome.skill == "order_pizza"
    engine.close()


def test_user_can_cancel_mid_dialogue(config, channel):
    channel.replies = ["large", "never mind"]
    engine = make_engine(config, channel, [PizzaSkill()])
    engine.handle("order a pizza")
    assert "Cancelled." in channel.said
    engine.close()


def test_a_crashing_skill_is_reported_not_propagated(config, channel):
    engine = make_engine(config, channel, [ExplodingSkill()])
    outcome = engine.handle("explode")
    assert not outcome.ok
    assert "went wrong" in channel.said
    engine.close()


def test_unmatched_utterance_says_so(config, channel):
    engine = make_engine(config, channel, [GreetSkill()])
    outcome = engine.handle("please recalibrate the flux capacitor")
    assert outcome.skill is None
    assert "don't know how to do that" in channel.said
    engine.close()


def test_history_is_recorded(config, channel):
    engine = make_engine(config, channel, [GreetSkill()])
    engine.handle("hello")
    rows = engine.store.recent_history(limit=5)
    assert rows[0]["skill"] == "greet"
    assert rows[0]["status"] == "ok"
    engine.close()


# -- the actual reference skill -------------------------------------------


@pytest.fixture
def dev_engine(config, channel, monkeypatch):
    """The real start_dev_environment skill with Cursor stubbed out."""
    from jarvis.skills import start_dev_environment as mod

    launched: list[list[str]] = []
    monkeypatch.setattr(mod, "which", lambda cmd: f"/usr/bin/{cmd}")
    monkeypatch.setattr(mod, "is_running", lambda pattern: True)
    monkeypatch.setattr(
        mod, "spawn_detached", lambda cmd, cwd=None: launched.append(cmd) or object()
    )
    monkeypatch.setattr(
        mod, "run_capture", lambda cmd, cwd=None, timeout=30: type(
            "R", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )()
    )

    engine = make_engine(config, channel, [mod.StartDevEnvironmentSkill()])
    engine.launched = launched
    yield engine
    engine.close()


def test_dev_environment_creates_a_new_project(dev_engine, channel, config):
    channel.replies = ["new", "invoice parser"]
    outcome = dev_engine.handle("start my dev environment")

    created = Path(config.projects.root) / "invoice-parser"
    assert created.is_dir(), "project directory should exist"
    assert (created / "README.md").exists()
    assert outcome.ok
    assert "invoice parser" in channel.said
    assert [str(created)] == [cmd[-1] for cmd in dev_engine.launched if len(cmd) > 1]


def test_dev_environment_opens_an_existing_project_by_name(dev_engine, channel, config):
    existing = Path(config.projects.root) / "portfolio-site"
    existing.mkdir()
    dev_engine.store.record_project(existing)

    channel.replies = ["existing", "portfolio site"]
    outcome = dev_engine.handle("start my dev environment")

    assert outcome.ok
    assert "Opened portfolio site in Cursor." in channel.said


def test_dev_environment_picks_a_project_by_number(dev_engine, channel, config):
    for name in ("alpha", "beta", "gamma"):
        path = Path(config.projects.root) / name
        path.mkdir()
        dev_engine.store.record_project(path)

    channel.replies = ["existing", "the second one"]
    dev_engine.handle("start my dev environment")

    # Most recently recorded first, so option two is beta.
    assert "Opened beta in Cursor." in channel.said


def test_dev_environment_offers_to_create_when_nothing_is_known(dev_engine, channel):
    channel.replies = ["existing", "yes", "scratchpad"]
    outcome = dev_engine.handle("start my dev environment")
    assert outcome.ok
    assert "scratchpad" in channel.said


def test_dev_environment_refuses_to_clobber_an_existing_folder(dev_engine, channel, config):
    (Path(config.projects.root) / "taken").mkdir()
    channel.replies = ["new", "taken", "no"]
    dev_engine.handle("start my dev environment")
    assert "already exists" in channel.said
    assert "leave it alone" in channel.said
