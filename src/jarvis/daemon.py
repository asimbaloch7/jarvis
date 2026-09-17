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
import threading
import time

import numpy as np

from .audio.input import AudioUnavailable, MicStream
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

    # -- lifecycle --------------------------------------------------------

    def stop(self, *_args) -> None:
        if not self._stop.is_set():
            log.info("Shutdown requested")
            self._stop.set()

    def _acquire_instance_lock(self) -> bool:
        """Keep a second copy from stealing the microphone.

        Two processes on the same device produce silent audio and missed
        wake words. The lock file is held for the life of this process.
        """
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
        except Exception as exc:
            log.error("Wake word engine failed to start: %s", exc)
            print_status(f"Wake word engine failed to start: {exc}", warn=True)
            self.tts.speak("I couldn't start my wake word detector. Check the log.")
            return 1

        try:
            self.mic.start()
        except AudioUnavailable as exc:
            log.error("%s", exc)
            print_status(str(exc), warn=True)
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
        # Load Whisper in the background so the first command isn't a multi-second stall.
        threading.Thread(target=self.stt.load, daemon=True, name="stt-preload").start()
        if self.greet:
            print_status("🗣️  Responding...")
            self.tts.speak("Jarvis online.")

        self._announce_idle()
        try:
            self._loop(channel)
        finally:
            self._shutdown()
        return 0

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
            chunk = self.mic.read(timeout=0.5)
            if chunk is None:
                self._maybe_unload()
                continue

            peak_rms = max(peak_rms, float(np.max(np.abs(chunk.astype(np.int32)))))

            try:
                score = self.wake.score(chunk)
            except Exception as exc:
                log.error("Wake detection error: %s", exc)
                time.sleep(0.5)
                continue

            peak_score = max(peak_score, score)
            now = time.monotonic()
            if now - last_heartbeat >= 8.0:
                if peak_rms < 200:
                    log.warning(
                        "Mic is nearly silent (peak %d / 32768). Unmute it, "
                        "or set audio.input_device in config.yaml (`jarvis devices`).",
                        int(peak_rms),
                    )
                else:
                    log.info(
                        "Still listening. Peak wake score %.2f (need %.2f). "
                        "Say 'Hey Jarvis' clearly toward the mic.",
                        peak_score,
                        threshold,
                    )
                peak_score = 0.0
                peak_rms = 0.0
                last_heartbeat = now

            if score < threshold or (now - last_trigger) < cooldown:
                self._maybe_unload()
                continue

            last_trigger = now
            self._last_activity = now
            log.info("Wake word detected (score %.2f)", score)
            print_status("✅ Wake word detected! Listening for your command...")
            pending = self.mic.drain()
            self.wake.reset()

            try:
                self._handle_wake(channel, pending)
            except Exception as exc:
                log.exception("Unhandled error while handling a command: %s", exc)
                print_status(f"Something went wrong: {exc}", warn=True)
                channel.cue("error")
                self.tts.speak("Something went wrong. It's in the log.")

            self._last_activity = time.monotonic()
            # Anything said during execution is not a command for us.
            self.mic.flush()
            self.wake.reset()
            self._announce_idle()

    def _handle_wake(self, channel: VoiceChannel, pending: list) -> None:
        # No spoken "Yes?" and no beep: both add lag, and playback often
        # mutes a Bluetooth mic so the command itself is lost.
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
