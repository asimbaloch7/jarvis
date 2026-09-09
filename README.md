# Jarvis

A local, always-listening voice assistant for Fedora Linux, with Google Gemini
as its reasoning layer and a plugin system for adding new voice commands.

Say **"Hey Jarvis"**, wait for the tone, then say what you want.

```
you>    Hey Jarvis
jarvis> (beep)
you>    Start my dev environment.
jarvis> Do you want to open an existing project, or start a new one?
you>    New.
jarvis> What should I call it?
you>    Invoice parser.
jarvis> Created invoice parser in Workspace, with git initialised,
        and opened it in Cursor.
```

## How it works

```
  microphone
      |
      v
 [ openWakeWord ]  always on, local, a few MB of RAM and ~2% of one core
      |  "hey jarvis"
      v
 [ record + VAD ]  stops when you stop talking
      |
      v
 [ faster-whisper ]  local transcription, loaded on demand, unloaded when idle
      |  "start my dev environment"
      v
 [ Gemini 2.5 Flash ]  function calling picks the skill and its arguments
      |  start_dev_environment(mode=None)      <-- falls back to keyword
      v                                            matching when offline
 [ skill runs ]  may ask follow-up questions, looping back to the recorder
      |
      v
 [ Piper TTS ]  spoken confirmation
```

Audio never leaves the machine before the wake word fires, and by default it
never leaves at all: transcription is local, and only the resulting text is
sent to Gemini.

## Quick start

From the repository root:

```bash
./scripts/setup.sh                  # dnf packages, venv, models
$EDITOR .env                        # paste your Gemini API key
source .venv/bin/activate
jarvis doctor                       # verify every stage
jarvis run                          # say "Hey Jarvis"
```

Every script resolves paths relative to the checkout, so it does not matter
where the repository lives.

Get a free API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

## Installation, step by step

`scripts/setup.sh` does all of this for you. Here is what it does and why, in
case you would rather do it by hand or something goes wrong.

### 1. System packages

```bash
sudo dnf install -y \
    python3.12 python3.12-devel \
    gcc gcc-c++ make \
    portaudio portaudio-devel \
    pipewire-utils alsa-utils \
    espeak-ng \
    git curl tar
```

**Why Python 3.12 and not your system Python?** `onnxruntime` (wake word) and
`ctranslate2` (Whisper) do not publish wheels for Python 3.13 or 3.14 yet. On
a current Fedora your default `python3` is too new, and pip will try to build
those from source and fail. Installing `python3.12` alongside it costs
nothing and changes nothing about your system Python.

`portaudio` is what `sounddevice` records through. `espeak-ng` is the TTS
fallback if Piper cannot be installed.

### 2. Virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
./scripts/install_python_deps.sh
```

The first install pulls a few hundred MB of wheels, mostly onnxruntime and
ctranslate2.

That script installs one package per `pip` invocation rather than running a
single `pip install -e .`, which matters on a slow or intermittent
connection. Pip's resolver backtracks through candidate versions and refetches
metadata for each one, so a single dropped request makes it give up on the
whole install — and it misreports the cause, claiming:

```
Additionally, some packages in these conflicts have no matching
distributions available for your environment:
    tokenizers
```

There is nothing wrong with `tokenizers`; that message means pip could not
fetch its index page. Installing one package at a time keeps each download
small and independently retried, and finished wheels stay in pip's cache, so
**re-running the script resumes rather than starting over**:

```bash
./scripts/install_python_deps.sh     # run as many times as it takes
```

`openwakeword` is installed with `--no-deps` for the same reason: on Linux
its metadata still requires `tflite-runtime`, which has no Python 3.12
wheel. Jarvis uses the ONNX backend (`onnxruntime` is already a dependency),
so TFLite is unused.

Raise the retry count if your link is especially bad:

```bash
JARVIS_PIP_ATTEMPTS=15 ./scripts/install_python_deps.sh
```

### 3. Models

```bash
./scripts/download_models.sh
```

This fetches three things:

| What | Where it goes | Size |
|---|---|---|
| openWakeWord `hey_jarvis_v0.1` + shared feature models | `~/.local/share/openwakeword` | ~20 MB |
| Piper binary and the `en_GB-alan-medium` voice | `~/.local/share/jarvis/piper`, symlinked to `~/.local/bin/piper` | ~85 MB |
| faster-whisper `base.en` | `~/.cache/huggingface` | ~75 MB |

To use a different voice, browse
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) and run
`JARVIS_VOICE=en_US-amy-medium ./scripts/download_models.sh`, then set
`tts.voice` in `config/config.yaml` to match.

### 4. API key

```bash
cp .env.example .env
$EDITOR .env        # GEMINI_API_KEY=...
```

`.env` and `config/config.yaml` are both gitignored. The key is only ever read
from the environment or that file, never from source.

### 5. Microphone

There is no special permission to grant on Fedora — any process in your user
session can record. What you do need is the right *default device*.

```bash
jarvis devices              # list everything PortAudio can see
wpctl status                # what PipeWire thinks is default
arecord -d 3 /tmp/t.wav && aplay /tmp/t.wav    # prove the hardware works
```

If the default is wrong, either set it in the GNOME Sound settings, or pin it
in `config/config.yaml`:

```yaml
audio:
  input_device: "HD Webcam"     # substring of the name, or the integer index
```

Check your input is not muted and the level is reasonable:

```bash
wpctl set-mute @DEFAULT_AUDIO_SOURCE@ 0
wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 1.0
```

## Testing each stage on its own

Debugging a wake-to-speech pipeline as one unit is miserable. Every stage has
its own subcommand, so you can find the broken one directly.

```bash
jarvis doctor                # checks all of the below and says what's missing
```

**Text to speech, no mic involved:**

```bash
jarvis say "All systems nominal."
jarvis beep wake             # the acknowledgement tone
```

**Wake word only.** Prints a line and beeps every time it fires. Leave it
running for a few minutes of normal conversation to check for false triggers.

```bash
jarvis wake
jarvis wake --show-scores    # live confidence values, for tuning the threshold
```

If it never fires, lower `wake.threshold` toward `0.4`. If it fires at random
noise, raise it toward `0.7`.

**Recording and transcription only.** No wake word, no Gemini.

```bash
jarvis listen -n 5
```

This also tells you the real transcription latency on your CPU. If `base.en`
feels slow, drop `stt.model` to `tiny.en`; if it mishears you often, try
`small.en` and accept a slower reply.

**Routing only.** Shows which skill Gemini picked and with what arguments,
without running anything.

```bash
jarvis route "start my dev environment"
jarvis route "how much disk space is left"
jarvis --offline route "open firefox"     # test the no-network path
```

**The whole pipeline, typed instead of spoken.** This is the fastest way to
develop a new skill, and it works over SSH with no audio at all.

```bash
jarvis text
```

**Everything, for real:**

```bash
jarvis run
```

Other useful commands:

```bash
jarvis skills                # every skill, its parameters, example phrasings
jarvis projects --rescan     # what Jarvis knows about your ~/Workspace
jarvis history               # what you asked for and how it was routed
tail -f ~/.local/state/jarvis/jarvis.log
```

## Running it in the background

```bash
./scripts/install_service.sh
```

That writes a `systemd --user` unit pointing at your checkout, enables it, and
turns on lingering so it survives logout.

```bash
systemctl --user status jarvis
systemctl --user restart jarvis     # after editing code or config
systemctl --user stop jarvis        # frees the microphone
journalctl --user -u jarvis -f      # live logs
```

It is a **user** service, not a system one. It needs your PipeWire session to
hear you and your desktop session to launch applications, neither of which a
system service has.

Idle cost is one Python process holding the wake-word model, at a couple of
percent of one core. Whisper loads on your first command and unloads again
after `general.idle_unload_seconds` (default five minutes) of quiet.

## Configuration

`config/config.yaml`, copied from `config.example.yaml`, which documents every
option. A partial file is fine; anything absent uses the built-in default.

Any value can also be overridden with an environment variable named
`JARVIS_<SECTION>_<KEY>`, which is handy for one-off experiments:

```bash
JARVIS_STT_MODEL=tiny.en jarvis listen
JARVIS_BRAIN_MODEL=gemini-2.5-pro jarvis text
JARVIS_LOG_LEVEL=DEBUG jarvis run
```

The settings you are most likely to touch:

| Setting | Default | What it does |
|---|---|---|
| `wake.threshold` | `0.5` | Raise for fewer false triggers, lower to be heard more easily |
| `stt.model` | `base.en` | `tiny.en` is faster, `small.en` is more accurate |
| `listen.silence_ms` | `800` | How long a pause ends your sentence |
| `brain.model` | `gemini-2.5-flash` | Any Gemini model with function calling |
| `projects.root` | `~/Workspace` | Where new projects are created and existing ones found |
| `general.offline_only` | `false` | Skip Gemini entirely and use keyword routing |

## When things go wrong

Jarvis is built to say what happened rather than fail silently.

- **No internet or Gemini is down.** It says so once, then routes with keyword
  matching. Simple commands keep working. `jarvis --offline text` simulates
  this on demand.
- **Bad or missing API key.** Same fallback, with a spoken message naming the
  problem.
- **Rate limited.** It tells you and stops, rather than retrying in a loop.
- **A skill crashes.** The traceback goes to the log, you hear a short error,
  and the daemon returns to listening.
- **Piper missing.** Falls back to espeak-ng, then to printing.
- **You go quiet mid-conversation.** The follow-up times out after
  `general.conversation_turn_timeout` seconds and Jarvis returns to idle.
- **You change your mind.** Say "cancel", "stop", or "never mind" at any point
  in a conversation.

Everything recognised, chosen, and failed is in
`~/.local/state/jarvis/jarvis.log` and in `jarvis history`.

## Writing a new skill

A skill is one file in `src/jarvis/skills/`. Drop it in and restart; discovery
is automatic. The `description` field is not documentation, it is the prompt
Gemini uses to decide when to call your skill, so write it for that purpose:
say when to use it *and when not to*.

```python
from .base import Ask, Skill, SkillContext, SkillResult


class TakeNoteSkill(Skill):
    name = "take_note"
    description = """
    Append a timestamped note to the user's notes file. Use this for "make a
    note", "remind me that", or "write this down". Do not use it for calendar
    events or reminders with a specific time.
    """
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The note, verbatim."}
        },
    }
    keywords = ["take a note", "make a note", "write this down"]
    examples = ["Take a note: the build breaks on Python 3.13."]

    def execute(self, params, ctx: SkillContext):
        text = params.get("text")
        if not text:
            text = yield Ask.free("What should the note say?")

        path = ctx.config.paths.data_path / "notes.md"
        with path.open("a") as handle:
            handle.write(f"- {text}\n")

        return SkillResult("Noted.", cue="done")
```

Then:

```bash
jarvis skills                       # confirm it was discovered
jarvis route "take a note"          # confirm Gemini picks it
jarvis text                         # try it end to end
```

### The interface

| Attribute | Required | Purpose |
|---|---|---|
| `name` | yes | The function name Gemini calls. snake_case. |
| `description` | yes | When to use this skill, and when not to. Written for Gemini. |
| `parameters` | no | JSON Schema. Every property needs a `description`. |
| `keywords` | no | Used by the offline router when Gemini is unreachable. |
| `examples` | no | Shown in `jarvis skills`. |
| `requires_confirmation` | no | Ask for a spoken yes before running at all. |
| `enabled` | no | Set `False` to disable without deleting the file. |
| `execute(params, ctx)` | yes | Return a `SkillResult`, or yield `Ask`s and then return one. |
| `extract_params(text)` | no | Pull arguments out of raw text for the offline path. |

### Asking questions

If `execute` contains a `yield`, it becomes a dialogue. Each `Ask` you yield is
spoken to the user, and their answer comes back as the value of the yield.

```python
size    = yield Ask.choice("Small or large?", ["small", "large"])   # -> str
name    = yield Ask.free("What should I call it?")                  # -> str
proceed = yield Ask.confirm("Shall I go ahead?")                    # -> bool
```

`choice` and `confirm` answers are resolved locally, so they cost no network
call and work offline. That covers the literal words, ordinals ("the second
one"), bare numbers, and near-misses from the transcriber. Only genuinely
ambiguous `choice` answers escalate to Gemini.

If the user cancels or stops replying, `ConversationCancelled` is thrown into
your generator, so `try/finally` cleanup works normally. You do not have to
handle it.

Two conventions worth following, because the output is spoken aloud: keep
`SkillResult.speech` to a sentence, and run names through something like
`.replace("-", " ")` so `invoice-parser` does not come out as
"invoice dash parser".

### Asking Cursor to write one

The skill format is deliberately small enough to describe in a sentence. In
Cursor, this works:

> Add a Jarvis skill in `src/jarvis/skills/` called `check_weather` that takes
> an optional city, defaults to my configured location, calls wttr.in, and
> speaks a one-sentence summary. Follow the pattern in
> `start_dev_environment.py`.

## Layout

```
src/jarvis/
  daemon.py       always-on loop: wake -> listen -> engine -> idle
  engine.py       routing plus the dialogue driver that pumps skill generators
  channels.py     voice vs text IO, so the engine never touches hardware
  cli.py          jarvis run | text | wake | listen | route | doctor | ...
  config.py       defaults, config.yaml, and JARVIS_* env overrides
  audio/          mic stream, VAD endpointing, playback, cue tones
  wake/           openWakeWord (default), Porcupine (optional)
  stt/            faster-whisper
  tts/            Piper, espeak-ng fallback
  brain/          Gemini function calling, prompts, offline answer resolution
  skills/         base.py, registry.py, and one file per capability
  state/          SQLite store and per-exchange conversation memory
```

Each external engine sits behind a small abstract base class, so swapping
Whisper for something else, or openWakeWord for Porcupine, is a config change
plus one new file rather than a refactor.

## Choices worth knowing about

**openWakeWord over Porcupine.** Fully open, no account, and it ships a
pretrained "hey jarvis" model. Porcupine uses less CPU and false-triggers
less, but needs a Picovoice key. `src/jarvis/wake/porcupine_engine.py` is
there if you want it: set `PICOVOICE_ACCESS_KEY` and `wake.engine: porcupine`.

**Local Whisper over cloud STT.** Your voice stays on the machine, and it
works without a network. The cost is latency: expect one to three seconds for
a short command with `base.en` on a laptop CPU.

**Gemini 2.5 Flash over Pro.** Function-call routing is an easy task, and
Flash answers fast enough that the conversation does not feel laggy. Change
`brain.model` if you disagree.

**Function calling over prompt-and-parse.** Gemini returns a structured call
with typed arguments, so there is no regex layer between the model and the
action, and adding a skill means adding a schema rather than a parser.

## License

MIT
