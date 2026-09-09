"""Command line entry point.

Beyond `jarvis run`, most of these subcommands exist to let you test one
stage of the pipeline at a time, which is far easier than debugging the whole
wake-to-speech chain at once.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import Config, load_config
from .logging_setup import setup_logging


def _load(args) -> Config:
    cfg = load_config(args.config)
    if getattr(args, "verbose", False):
        cfg.paths.log_level = "DEBUG"
    if getattr(args, "offline", False):
        cfg.general.offline_only = True
    cfg.ensure_dirs()
    setup_logging(cfg)
    return cfg


# -- commands -------------------------------------------------------------


def cmd_run(args) -> int:
    from .daemon import run_daemon

    cfg = _load(args)
    return run_daemon(cfg, greet=not args.quiet)


def cmd_text(args) -> int:
    """Full pipeline minus the audio: type commands, read replies."""
    from .channels import TextChannel
    from .engine import Engine

    cfg = _load(args)
    engine = Engine(cfg, TextChannel())
    engine.store.scan_projects(cfg.projects.root_path, cfg.projects.scan_depth)

    mode = "gemini: " + cfg.brain.model if engine.router.gemini_enabled else "offline keyword mode"
    print(f"\nJarvis text mode ({len(engine.registry)} skills, {mode}).")
    print("Type a command, or 'quit' to exit.\n")

    try:
        while True:
            try:
                line = input("  you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.lower() in ("quit", "exit", ":q"):
                break
            engine.handle(line)
    finally:
        engine.close()
    return 0


def cmd_say(args) -> int:
    """Stage test: TTS only."""
    from .audio.output import Speaker
    from .tts.base import create_tts_engine

    cfg = _load(args)
    speaker = Speaker(cfg.audio.output_device)
    tts = create_tts_engine(cfg.tts, speaker)
    print(f"Engine: {type(tts).__name__}")
    tts.speak(args.text)
    return 0


def cmd_beep(args) -> int:
    from .audio.output import Speaker

    cfg = _load(args)
    Speaker(cfg.audio.output_device).beep(args.kind)
    return 0


def cmd_wake(args) -> int:
    """Stage test: wake word only. Prints a line each time it fires."""
    from .audio.input import AudioUnavailable, MicStream
    from .audio.output import Speaker
    from .wake.base import create_wake_detector

    cfg = _load(args)
    detector = create_wake_detector(cfg.wake)
    speaker = Speaker(cfg.audio.output_device)

    print(f"Listening for '{cfg.wake.model}' (threshold {cfg.wake.threshold}). Ctrl-C to stop.")
    if args.show_scores:
        print("Showing live scores. Speak the wake word and watch the peak.\n")

    detections = 0
    peak = 0.0
    try:
        with MicStream(cfg.audio.input_device, cfg.audio.sample_rate) as mic:
            last_fire = 0.0
            while True:
                chunk = mic.read(timeout=1.0)
                if chunk is None:
                    continue
                score = detector.score(chunk)
                peak = max(peak, score)
                if args.show_scores and score > 0.01:
                    print(f"  score {score:.3f}")
                now = time.monotonic()
                if score >= cfg.wake.threshold and now - last_fire > cfg.wake.cooldown_seconds:
                    last_fire = now
                    detections += 1
                    print(f"  DETECTED  #{detections}  (score {score:.3f})")
                    speaker.beep("wake")
                    detector.reset()
    except AudioUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"\n{detections} detection(s). Peak score {peak:.3f}.")
    return 0


def cmd_listen(args) -> int:
    """Stage test: record + transcribe, no wake word, no routing."""
    from .audio.input import AudioUnavailable, MicStream
    from .audio.output import Speaker
    from .audio.recorder import record_utterance
    from .stt.base import create_stt_engine

    cfg = _load(args)
    stt = create_stt_engine(cfg.stt)
    print(f"Loading {cfg.stt.model}...")
    stt.load()
    speaker = Speaker(cfg.audio.output_device)

    try:
        with MicStream(cfg.audio.input_device, cfg.audio.sample_rate) as mic:
            for i in range(args.count):
                print(f"\n[{i + 1}/{args.count}] Speak after the beep.")
                speaker.beep("wake")
                utterance = record_utterance(mic, cfg.listen)
                if not utterance.has_speech:
                    print("  (heard nothing)")
                    continue
                started = time.monotonic()
                text = stt.transcribe(utterance.audio)
                print(f"  {utterance.duration:.1f}s audio -> {time.monotonic() - started:.1f}s")
                print(f"  transcript: {text!r}")
    except AudioUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print()
    return 0


def cmd_route(args) -> int:
    """Stage test: routing only. Shows the skill and params, runs nothing."""
    from .channels import TextChannel
    from .engine import Engine

    cfg = _load(args)
    engine = Engine(cfg, TextChannel())
    decision = engine.router.route(args.text, engine.conversation)
    print(f"  source: {decision.source}{' (degraded)' if decision.degraded else ''}")
    print(f"  skill:  {decision.skill.name if decision.skill else '-'}")
    print(f"  params: {decision.params}")
    if decision.reply:
        print(f"  reply:  {decision.reply}")
    engine.close()
    return 0


def cmd_skills(args) -> int:
    from .skills.registry import build_registry

    _load(args)
    registry = build_registry()
    for skill in registry.all():
        print(f"\n{skill.name}")
        print(f"  {' '.join(skill.description.split())[:300]}")
        props = (skill.parameters or {}).get("properties", {})
        required = set((skill.parameters or {}).get("required", []))
        if props:
            print("  parameters:")
            for key, spec in props.items():
                flag = "required" if key in required else "optional"
                enum = f" one of {spec['enum']}" if "enum" in spec else ""
                print(f"    - {key} ({spec.get('type', 'any')}, {flag}){enum}")
        if skill.examples:
            print(f"  examples: {'; '.join(skill.examples)}")
    print()
    return 0


def cmd_projects(args) -> int:
    from .state.store import Store

    cfg = _load(args)
    store = Store(cfg.paths.db_path)
    if args.rescan:
        found = store.scan_projects(cfg.projects.root_path, cfg.projects.scan_depth)
        print(f"Scanned {cfg.projects.root}: {found} new project(s).")
    projects = store.recent_projects(limit=args.limit)
    if not projects:
        print(f"No projects known yet. Looked under {cfg.projects.root}.")
    for index, project in enumerate(projects, 1):
        when = (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(project.last_opened))
            if project.last_opened
            else "never"
        )
        print(f"  {index:2}. {project.name:<28} {when:<17} {project.path}")
    store.close()
    return 0


def cmd_history(args) -> int:
    from .state.store import Store

    cfg = _load(args)
    store = Store(cfg.paths.db_path)
    for row in reversed(store.recent_history(limit=args.limit)):
        when = time.strftime("%m-%d %H:%M", time.localtime(row["ts"]))
        print(f"  {when}  {row['status']:<9} {row['skill'] or '-':<24} {row['transcript']}")
    store.close()
    return 0


def cmd_devices(args) -> int:
    from .audio.input import AudioUnavailable, list_devices

    _load(args)
    try:
        devices = list_devices()
    except AudioUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("  idx  in  out  name")
    for index, device in enumerate(devices):
        print(
            f"  {index:3}  {device['max_input_channels']:2}  "
            f"{device['max_output_channels']:3}  {device['name']}"
        )
    return 0


def cmd_doctor(args) -> int:
    """Check every dependency and report what still needs doing."""
    cfg = _load(args)
    problems: list[str] = []

    def check(
        label: str, ok: bool, detail: str = "", fix: str = "", fatal: bool = True
    ) -> None:
        """`detail` is always shown; `fix` only when the check failed."""
        mark = "ok  " if ok else ("FAIL" if fatal else "warn")
        note = detail if ok else " ".join(filter(None, [detail, fix]))
        print(f"  [{mark}] {label}" + (f"  -- {note}" if note else ""))
        if not ok and fatal:
            problems.append(label)

    print("\nPython")
    version = sys.version_info
    check(
        f"interpreter {version.major}.{version.minor}.{version.micro}",
        (3, 10) <= (version.major, version.minor) < (3, 13),
        fix="need 3.10-3.12; onnxruntime and ctranslate2 lack 3.13+ wheels. "
        "Rebuild the venv with: python3.12 -m venv .venv",
    )

    print("\nPackages")
    for module, hint in [
        ("numpy", "pip install numpy"),
        ("sounddevice", "sudo dnf install portaudio && pip install sounddevice"),
        ("openwakeword", "run ./scripts/install_python_deps.sh (installs it without tflite-runtime)"),
        ("onnxruntime", "pip install onnxruntime"),
        ("faster_whisper", "pip install faster-whisper"),
        ("google.genai", "pip install google-genai"),
        ("yaml", "pip install PyYAML"),
    ]:
        try:
            __import__(module)
            check(module, True)
        except Exception as exc:
            check(module, False, f"{type(exc).__name__}: {exc}.", f"Try: {hint}")

    print("\nAudio")
    try:
        from .audio.input import list_devices

        devices = list_devices()
        inputs = [d for d in devices if d["max_input_channels"] > 0]
        outputs = [d for d in devices if d["max_output_channels"] > 0]
        check(f"{len(inputs)} input device(s)", bool(inputs), fix="run: jarvis devices")
        check(f"{len(outputs)} output device(s)", bool(outputs), fix="run: jarvis devices")
    except Exception as exc:
        check("audio devices", False, str(exc))

    print("\nWake word")
    try:
        from .wake.base import create_wake_detector

        create_wake_detector(cfg.wake)
        check(f"model {cfg.wake.model}", True)
    except Exception as exc:
        check(f"model {cfg.wake.model}", False, str(exc))

    print("\nSpeech to text")
    try:
        from .stt.base import create_stt_engine

        engine = create_stt_engine(cfg.stt)
        started = time.monotonic()
        engine.load()
        check(f"whisper {cfg.stt.model}", True, f"loaded in {time.monotonic() - started:.1f}s")
        engine.unload()
    except Exception as exc:
        check(f"whisper {cfg.stt.model}", False, str(exc))

    print("\nText to speech")
    try:
        from .audio.output import Speaker
        from .tts.base import NullTts, create_tts_engine

        tts = create_tts_engine(cfg.tts, Speaker(cfg.audio.output_device))
        actual = type(tts).__name__.replace("Tts", "").lower()
        # Name the engine actually in use, which may not be the configured one.
        label = (
            f"{actual} ({cfg.tts.voice})" if actual == "piper" else f"{actual}"
        )
        if actual != cfg.tts.engine.lower() and not isinstance(tts, NullTts):
            label += f", configured as {cfg.tts.engine}"
        check(
            label,
            not isinstance(tts, NullTts),
            fix="run ./scripts/download_models.sh, or set tts.engine: espeak",
            fatal=False,
        )
    except Exception as exc:
        check("tts", False, str(exc), fatal=False)

    print("\nGemini")
    if not cfg.gemini_api_key:
        check(
            "GEMINI_API_KEY",
            False,
            fix="copy .env.example to .env and set the key",
            fatal=False,
        )
    else:
        check("GEMINI_API_KEY", True, f"...{cfg.gemini_api_key[-4:]}")
        try:
            from .brain.gemini import GeminiBrain

            brain = GeminiBrain(cfg.brain, cfg.gemini_api_key)
            reply = brain.complete("Reply with the single word: ready")
            check(
                f"live call to {cfg.brain.model}",
                "ready" in reply.lower(),
                repr(reply[:60]),
                fatal=False,
            )
        except Exception as exc:
            check(f"live call to {cfg.brain.model}", False, str(exc)[:160], fatal=False)

    print("\nSkills and paths")
    try:
        from .skills.registry import build_registry

        registry = build_registry()
        check(f"{len(registry)} skill(s) loaded", len(registry) > 0)
    except Exception as exc:
        check("skill registry", False, str(exc))

    from .util.process import which

    editor = cfg.projects.editor_command
    check(
        f"editor '{editor}' on PATH",
        which(editor) is not None,
        fix="install it, or set projects.editor_command to something on your PATH",
        fatal=False,
    )
    check("git on PATH", which("git") is not None, fix="sudo dnf install git", fatal=False)
    check(
        f"projects root {cfg.projects.root}",
        cfg.projects.root_path.is_dir(),
        fix="create it, or change projects.root in config.yaml",
        fatal=False,
    )
    check(f"log file {cfg.paths.log_file}", cfg.paths.log_path.parent.is_dir())

    if problems:
        print(f"\n{len(problems)} blocking problem(s): {', '.join(problems)}\n")
        return 1
    print("\nAll clear. Try: jarvis text\n")
    return 0


# -- parser ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="A local voice assistant with a Gemini brain.",
    )
    parser.add_argument("-c", "--config", type=Path, help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument(
        "--offline", action="store_true", help="skip Gemini, use keyword routing only"
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="start the always-listening daemon")
    run.add_argument("--quiet", action="store_true", help="no spoken greeting on start")
    run.set_defaults(func=cmd_run)

    text = sub.add_parser("text", help="type commands instead of speaking (no audio needed)")
    text.set_defaults(func=cmd_text)

    say = sub.add_parser("say", help="stage test: speak a phrase")
    say.add_argument("text", nargs="?", default="Systems are online and ready.")
    say.set_defaults(func=cmd_say)

    beep = sub.add_parser("beep", help="stage test: play a cue tone")
    beep.add_argument("kind", nargs="?", default="wake", choices=["wake", "done", "error"])
    beep.set_defaults(func=cmd_beep)

    wake = sub.add_parser("wake", help="stage test: wake word detection only")
    wake.add_argument("--show-scores", action="store_true", help="print live confidence")
    wake.set_defaults(func=cmd_wake)

    listen = sub.add_parser("listen", help="stage test: record and transcribe")
    listen.add_argument("-n", "--count", type=int, default=3, help="how many utterances")
    listen.set_defaults(func=cmd_listen)

    route = sub.add_parser("route", help="stage test: show routing for a phrase, run nothing")
    route.add_argument("text")
    route.set_defaults(func=cmd_route)

    skills = sub.add_parser("skills", help="list installed skills and their parameters")
    skills.set_defaults(func=cmd_skills)

    projects = sub.add_parser("projects", help="show known projects")
    projects.add_argument("--rescan", action="store_true", help="rescan the projects root")
    projects.add_argument("-n", "--limit", type=int, default=15)
    projects.set_defaults(func=cmd_projects)

    history = sub.add_parser("history", help="recent commands and how they were routed")
    history.add_argument("-n", "--limit", type=int, default=25)
    history.set_defaults(func=cmd_history)

    devices = sub.add_parser("devices", help="list audio devices")
    devices.set_defaults(func=cmd_devices)

    doctor = sub.add_parser("doctor", help="check the whole setup and report problems")
    doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except RuntimeError as exc:
        # Setup problems (unwritable paths, missing engines) are expected
        # failure modes, not bugs. Say what's wrong without a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
