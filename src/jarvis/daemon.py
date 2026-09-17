"""The always-on background process.

Idle cost is deliberately small: only the wake-word model is resident and
running. Whisper is loaded on the first command and unloaded again after
`general.idle_unload_seconds` of quiet, so a Jarvis that has been sitting
there all afternoon is roughly as cheap as one that just started.
"""

from __future__ import annotations

import fcntl
import os
import signal
import sys
import threading
import time

import numpy as np

from .audio.input import (
    AudioUnavailable,
    MicStream,
    capture_devices,
    output_only_headsets,
)
from .audio.output import Speaker
from .brain.answers import strip_wake_phrase
from .channels import VoiceChannel
from .config import Config
from .engine import Engine
from .logging_setup import get_logger, print_status
from .state.store import Store
from .stt.base import create_stt_engine
from .tts.base import create_tts_engine
from .wake.base import create_wake_detector

log = get_logger("daemon")


class JarvisDaemon:
    def __init__(self, cfg: Config, greet: bool = True):
        self.cfg = cfg
        self.greet = greet
        self._stop = threading.Event()
        self._manual_wake = threading.Event()
        self._last_activity = time.monotonic()

        cfg.ensure_dirs()
        self.store = Store(cfg.paths.db_path)
        self.speaker = Speaker(cfg.audio.output_device)
        self.tts = create_tts_engine(cfg.tts, self.speaker)
        self.stt = create_stt_engine(cfg.stt)
        self.mic = MicStream(cfg.audio.input_device, cfg.audio.sample_rate)
        self.wake = None  # loaded in run(), so failures are reportable
        self.engine: Engine | None = None
        self._lock_file = None
        self._silent_warned_at = 0.0

    # -- lifecycle --------------------------------------------------------

    def stop(self, *_args) -> None:
        if not self._stop.is_set():
            log.info("Shutdown requested")
            self._stop.set()

    def _acquire_instance_lock(self) -> bool:
        """Keep a second copy from stealing the microphone."""
        path = self.cfg.paths.log_path.parent / "jarvis.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            other = fh.read().strip() or "unknown"
            fh.close()
            print_status(
                f"Jarvis is already running (pid {other}). "
                "Two copies fight over the mic, so neither hears you.",
                warn=True,
            )
            print_status("Stop the other one first:  systemctl --user stop jarvis", warn=True)
            log.error("Another Jarvis is already running (pid %s)", other)
            return False
        fh.seek(0)
        fh.truncate()
        fh.write(str(os.getpid()))
        fh.flush()
        self._lock_file = fh
        return True

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self.stop)
            except ValueError:
                pass  # not on the main thread, e.g. under a test runner

    def _watch_enter_key(self) -> None:
        """Enter-to-talk, so a missed wake word is never a dead end."""
        if not sys.stdin.isatty():
            return

        def _loop() -> None:
            while not self._stop.is_set():
                line = sys.stdin.readline()
                if line == "":
                    return
                self._manual_wake.set()

        threading.Thread(target=_loop, daemon=True, name="enter-to-talk").start()

    def run(self) -> int:
        self._install_signal_handlers()
        if not self._acquire_instance_lock():
            return 1

        seeded = self.store.scan_projects(
            self.cfg.projects.root_path, self.cfg.projects.scan_depth
        )
        if seeded:
            log.info("Seeded %d project(s) from %s", seeded, self.cfg.projects.root)

        try:
            self.wake = create_wake_detector(self.cfg.wake)
        except Exception as extra:
            log.error("Wake word engine failed to start: %s", extra)
            print_status(f"Wake word engine failed to start: {extra}", warn=True)
            self.tts.speak("I couldn't start my wake word detector. Check the log.")
            return 1

        self._announce_devices()

        try:
            self.mic.start()
        except AudioUnavailable as extra:
            log.error("%s", extra)
            print_status(str(extra), warn=True)
            self.tts.speak("I can't open the microphone, so I can't listen.")
            return 1

        channel = VoiceChannel(self.cfg, self.mic, self.stt, self.tts, self.speaker)
        self.engine = Engine(self.cfg, channel, store=self.store)

        log.info(
            "Jarvis is listening for the wake word (%d skills, brain=%s)",
            len(self.engine.registry),
            self.cfg.brain.model if self.engine.router.gemini_enabled else "offline",
        )
        print_status(f"Mic: {self.mic.device_label}")
        print_status('Say "Hey Jarvis" then your command — or press Enter to talk.')
        threading.Thread(target=self.stt.load, daemon=True, name="stt-preload").start()
        self._watch_enter_key()
        if self.greet:
            print_status("🗣️  Responding...")
            self.tts.speak("Jarvis online.")

        self._announce_idle()
        try:
            self._loop(channel)
        finally:
            self._shutdown()
        return 0

    def _announce_devices(self) -> None:
        try:
            devices = capture_devices()
        except AudioUnavailable:
            return
        if not devices:
            print_status("No capture devices found.", warn=True)
            return
        print_status("Capture devices:")
        for idx, dev in devices:
            n_in = int(dev["max_input_channels"])
            print_status(f"  {idx:>3}  {n_in}ch  {dev['name']}")
        for name in output_only_headsets():
            print_status(
                f"{name!r} is speakers-only (no mic). "
                "Enable Headset / Hands-Free mode in Sound settings, then restart.",
                warn=True,
            )

    def _announce_idle(self) -> None:
        print_status('🎙️  Listening for "Hey Jarvis"...')

    def _shutdown(self) -> None:
        print_status("Stopped.")
        log.info("Shutting down")
        self.mic.stop()
        if self.wake is not None:
            self.wake.close()
        self.stt.unload()
        self.tts.close()
        if self.engine is not None:
            self.engine.close()
        else:
            self.store.close()

    # -- main loop --------------------------------------------------------

    def _loop(self, channel: VoiceChannel) -> None:
        threshold = self.cfg.wake.threshold
        cooldown = self.cfg.wake.cooldown_seconds
        last_trigger = 0.0
        peak_score = 0.0
        peak_rms = 0.0
        last_heartbeat = time.monotonic()

        while not self._stop.is_set():
            manual = self._manual_wake.is_set()
            if manual:
                self._manual_wake.clear()

            chunk = self.mic.read(timeout=0.5)
            now = time.monotonic()

            if chunk is not None:
                peak_rms = max(peak_rms, float(np.max(np.abs(chunk.astype(np.int32)))))
                try:
                    score = self.wake.score(chunk)
                except Exception as extra:
                    log.error("Wake detection error: %s", extra)
                    time.sleep(0.5)
                    continue
                peak_score = max(peak_score, score)
            else:
                score = 0.0

            if now - last_heartbeat >= 4.0:
                self._heartbeat(now, peak_rms, peak_score, threshold)
                peak_score = 0.0
                peak_rms = 0.0
                last_heartbeat = now

            if chunk is None and not manual:
                self._maybe_unload()
                continue

            fired = manual or (chunk is not None and score >= threshold)
            if not fired or (now - last_trigger) < cooldown:
                self._maybe_unload()
                continue

            last_trigger = now
            self._last_activity = now
            if manual:
                log.info("Manual wake (Enter)")
                print_status("✅ Listening for your command... (Enter)")
            else:
                log.info("Wake word detected (score %.2f)", score)
                print_status("✅ Wake word detected! Listening for your command...")
            pending = self.mic.drain() if chunk is not None else []
            self.wake.reset()

            try:
                self._handle_wake(channel, pending)
            except Exception as extra:
                log.exception("Unhandled error while handling a command: %s", extra)
                print_status(f"Something went wrong: {extra}", warn=True)
                channel.cue("error")
                self.tts.speak("Something went wrong. It's in the log.")

            self._last_activity = time.monotonic()
            self.mic.flush()
            self.wake.reset()
            self._announce_idle()

    def _heartbeat(
        self, now: float, peak_rms: float, peak_score: float, threshold: float
    ) -> None:
        mic_pct = min(100, int(peak_rms / 327.68))
        print_status(
            f"Listening…  mic {mic_pct:3d}%  wake {peak_score:.2f}/{threshold:.2f}"
        )
        if peak_rms < 200:
            if now - self._silent_warned_at >= 12:
                print_status(
                    f"Mic is silent on {self.mic.device_label}. "
                    "Unmute it, or set audio.input_device from `jarvis devices`.",
                    warn=True,
                )
                self._silent_warned_at = now
        else:
            log.info(
                "Still listening. Peak wake score %.2f (need %.2f).",
                peak_score,
                threshold,
            )

    def _handle_wake(self, channel: VoiceChannel, pending: list) -> None:
        # No spoken "Yes?" — that adds lag and often mutes a Bluetooth mic.
        text = channel.listen(preroll=pending, flush=False)
        text = strip_wake_phrase(text or "")

        if not text:
            log.info("Wake word fired but nothing was said")
            print_status("No speech heard after the wake word.", warn=True)
            channel.speak("I didn't catch a command.")
            return

        self.engine.handle(text)

    def _maybe_unload(self) -> None:
        """Drop the Whisper model once we've been idle long enough."""
        timeout = self.cfg.general.idle_unload_seconds
        if timeout <= 0 or not self.stt.is_loaded:
            return
        if time.monotonic() - self._last_activity > timeout:
            log.info("Idle for %ds, unloading the STT model", timeout)
            self.stt.unload()


def run_daemon(cfg: Config, greet: bool = True) -> int:
    return JarvisDaemon(cfg, greet=greet).run()
