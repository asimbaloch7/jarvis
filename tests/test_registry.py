"""Skill discovery and the schema handed to Gemini."""

from __future__ import annotations

from jarvis.skills.registry import build_registry


def test_the_three_starting_skills_are_discovered():
    registry = build_registry()
    names = {skill.name for skill in registry.all()}
    assert {"start_dev_environment", "open_application", "run_shell_command"} <= names


def test_every_skill_produces_a_valid_function_declaration():
    for declaration in build_registry().function_declarations():
        assert declaration["name"] and declaration["name"].islower()
        assert len(declaration["description"]) > 40, "Gemini needs a real description"
        params = declaration["parameters"]
        assert params["type"] == "object"
        for spec in params.get("properties", {}).values():
            assert "type" in spec
            assert "description" in spec, "every parameter needs a description"
        for required in params.get("required", []):
            assert required in params["properties"]


def test_keyword_routing_finds_the_dev_environment_skill():
    registry = build_registry()
    match = registry.best_match("start my dev environment")
    assert match is not None
    assert match[0].name == "start_dev_environment"


def test_keyword_routing_declines_nonsense():
    assert build_registry().best_match("what is the airspeed velocity of a swallow") is None


def test_offline_parameter_extraction():
    registry = build_registry()
    dev = registry.get("start_dev_environment")
    assert dev.extract_params("create a new project called widgets")["mode"] == "new"
    assert "widgets" in dev.extract_params("create a new project called widgets")["project"]

    shell = registry.get("run_shell_command")
    assert shell.extract_params("run the command df -h")["command"] == "df -h"
