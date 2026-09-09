"""Run a shell command, with a safety gate in front of it.

Three tiers:
  denylist  -> refused outright, no confirmation offered
  allowlist -> runs immediately, as long as it needs no shell interpretation
  anything else -> read back verbatim and run only on a spoken "yes"

Shell metacharacters always force confirmation, even for an allowlisted
program, because `git log | tee /etc/passwd` starts with an allowlisted word.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from ..logging_setup import get_logger
from .base import Ask, Skill, SkillContext, SkillResult

log = get_logger("skill.run_shell_command")

SHELL_METACHARS = re.compile(r"[|;&><`$]|\$\(")
MAX_SPOKEN_CHARS = 240


class RunShellCommandSkill(Skill):
    name = "run_shell_command"
    description = """
    Run a shell command on the user's Fedora Linux machine and report the
    result out loud. Use this for system queries and one-off terminal work:
    disk usage, uptime, package queries, git status, systemctl status, and
    similar.

    Prefer a more specific skill when one exists. Never use this to open an
    application (use open_application) or to open a project (use
    start_dev_environment). Pass a single complete command line. The user will
    be asked to confirm anything that is not a plain read-only command, so do
    not worry about asking permission yourself.
    """
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The complete shell command to run, e.g. 'df -h /home'.",
            },
            "purpose": {
                "type": "string",
                "description": (
                    "Short plain-English description of what this achieves, "
                    "spoken to the user when confirming."
                ),
            },
        },
        "required": ["command"],
    }
    keywords = ["run command", "run the command", "terminal", "shell", "execute"]
    examples = [
        "How much disk space do I have left?",
        "Run git status.",
        "What's my uptime?",
    ]

    def execute(self, params: dict[str, Any], ctx: SkillContext):
        command = (params.get("command") or "").strip()
        if not command:
            return SkillResult.error("No command was given.")

        blocked = self._denied(command, ctx)
        if blocked:
            log.warning("Refused command %r (matched %s)", command, blocked)
            return SkillResult(
                "That command is on my blocked list, so I won't run it.",
                ok=False,
                data={"command": command, "matched": blocked},
                cue="error",
                status="refused",
            )

        needs_shell = bool(SHELL_METACHARS.search(command))
        if not self._allowlisted(command, ctx) or needs_shell:
            reason = (
                " It uses shell redirection or piping." if needs_shell else ""
            )
            approved = yield Ask.confirm(
                f"I'm about to run: {self._speakable(command)}.{reason} Should I?",
                reprompt=f"Yes or no. Run {self._speakable(command)}?",
            )
            if not approved:
                return SkillResult(
                    "Cancelled, I didn't run it.",
                    data={"command": command},
                    cue="done",
                    status="declined",
                )

        return self._run(command, needs_shell, ctx)

    # -- safety -----------------------------------------------------------

    @staticmethod
    def _denied(command: str, ctx: SkillContext) -> str | None:
        for pattern in ctx.config.shell.denylist_patterns:
            try:
                if re.search(pattern, command, re.IGNORECASE):
                    return pattern
            except re.error:
                log.error("Invalid denylist regex in config: %s", pattern)
        return None

    @staticmethod
    def _allowlisted(command: str, ctx: SkillContext) -> bool:
        try:
            parts = shlex.split(command)
        except ValueError:
            return False
        if not parts:
            return False
        program = Path(parts[0]).name
        return program in ctx.config.shell.allowlist

    # -- execution --------------------------------------------------------

    def _run(self, command: str, needs_shell: bool, ctx: SkillContext) -> SkillResult:
        cwd = Path.home()
        try:
            if needs_shell:
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True,
                    timeout=30, cwd=cwd,
                )
            else:
                result = subprocess.run(
                    shlex.split(command), capture_output=True, text=True,
                    timeout=30, cwd=cwd,
                )
        except subprocess.TimeoutExpired:
            return SkillResult.error("That command took too long, so I stopped it.")
        except FileNotFoundError:
            return SkillResult.error(f"There's no command called {command.split()[0]}.")
        except (OSError, ValueError) as exc:
            return SkillResult.error(f"I couldn't run that. {exc}")

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        log.info("Ran %r -> rc=%d, %d bytes out", command, result.returncode, len(stdout))

        if result.returncode != 0:
            detail = self._summarize(stderr or stdout) or f"exit code {result.returncode}"
            return SkillResult(
                f"That failed. {detail}",
                ok=False,
                data={"command": command, "returncode": result.returncode},
                cue="error",
            )

        if not stdout:
            return SkillResult("Done. No output.", data={"command": command}, cue="done")

        return SkillResult(
            self._summarize(stdout),
            data={"command": command, "output": stdout[:4000]},
            cue="done",
        )

    @staticmethod
    def _summarize(output: str) -> str:
        """Make command output listenable rather than dumping a screenful."""
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return ""
        if len(lines) == 1 and len(lines[0]) <= MAX_SPOKEN_CHARS:
            return lines[0]

        spoken = ""
        for line in lines:
            if len(spoken) + len(line) > MAX_SPOKEN_CHARS:
                break
            spoken += line + ". "
        spoken = spoken.strip()
        remaining = len(lines) - spoken.count(". ")
        if remaining > 0:
            spoken += f" And {remaining} more lines, which are in the log."
        return spoken

    @staticmethod
    def _speakable(command: str) -> str:
        """Read symbols aloud so a confirmation is unambiguous."""
        replacements = {
            "|": " pipe ",
            ">>": " append to ",
            ">": " redirect to ",
            "&&": " and then ",
            ";": " then ",
            "-rf": " dash r f ",
            "/": " slash ",
        }
        spoken = command
        for symbol, words in replacements.items():
            spoken = spoken.replace(symbol, words)
        return re.sub(r"\s+", " ", spoken).strip()

    def extract_params(self, text: str) -> dict[str, Any]:
        lowered = text.lower()
        for marker in ("run the command ", "run command ", "execute ", "run "):
            if marker in lowered:
                return {"command": text[lowered.index(marker) + len(marker):].strip()}
        return {}
