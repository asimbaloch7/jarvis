#!/usr/bin/env bash
# Fedora setup: system packages, a Python 3.12 venv, and local config files.
# Safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$1"; }

# ---------------------------------------------------------------- packages
say "Installing system packages with dnf"
# python3.12: onnxruntime and ctranslate2 have no wheels for 3.13+ yet.
# portaudio: the backend sounddevice records through.
# espeak-ng: TTS fallback if Piper is unavailable.
sudo dnf install -y \
    python3.12 python3.12-devel \
    gcc gcc-c++ make \
    portaudio portaudio-devel \
    pipewire-utils alsa-utils \
    espeak-ng \
    git curl tar

# ------------------------------------------------------------------- venv
if [[ ! -d .venv ]]; then
    say "Creating a Python 3.12 virtual environment"
    python3.12 -m venv .venv
else
    say "Reusing the existing .venv"
fi

VENV_PY_VERSION="$(.venv/bin/python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
if [[ "$VENV_PY_VERSION" == "3.13" || "$VENV_PY_VERSION" == "3.14" ]]; then
    warn ".venv is Python ${VENV_PY_VERSION}, which has no onnxruntime wheels. Rebuilding with 3.12."
    rm -rf .venv
    python3.12 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

say "Installing Python dependencies (a few hundred MB the first time)"
# Delegated to its own script so you can re-run just this part after a dropped
# connection, without redoing the dnf and venv steps above.
if ! ./scripts/install_python_deps.sh; then
    warn "Dependency install incomplete. Re-run ./scripts/install_python_deps.sh to resume."
    exit 1
fi

# ----------------------------------------------------------------- config
say "Setting up local config"
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo "  created .env -- put your Gemini API key in it"
else
    echo "  .env already exists, leaving it alone"
fi

if [[ ! -f config/config.yaml ]]; then
    cp config/config.example.yaml config/config.yaml
    echo "  created config/config.yaml"
else
    echo "  config/config.yaml already exists, leaving it alone"
fi

mkdir -p ~/.local/share/jarvis ~/.local/state/jarvis

# ----------------------------------------------------------------- models
say "Downloading models"
./scripts/download_models.sh

# ------------------------------------------------------------------- done
say "Done"
cat <<'EOF'

Next steps:

  1. Put your Gemini API key in .env       https://aistudio.google.com/apikey
  2. source .venv/bin/activate   # not `./.venv/bin/activate`
  3. jarvis doctor        # verify every stage
  4. jarvis text          # try the pipeline with no microphone
  5. jarvis run           # the real thing: say "Hey Jarvis"

To start it automatically on login:

  ./scripts/install_service.sh

EOF
