#!/usr/bin/env bash
# Fetch the wake-word models, the Piper binary, and a Piper voice.
# Everything lands under ~/.local/share/jarvis. Safe to re-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${HOME}/.local/share/jarvis"
PIPER_DIR="${DATA_DIR}/piper"
BIN_DIR="${HOME}/.local/bin"

VOICE="${JARVIS_VOICE:-en_GB-alan-medium}"
PIPER_VERSION="2023.11.14-2"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$1"; }

mkdir -p "$PIPER_DIR" "$BIN_DIR"

# ------------------------------------------------------- wake word models
say "Downloading openWakeWord models"
PYTHON="python3"
[[ -x "${REPO_ROOT}/.venv/bin/python" ]] && PYTHON="${REPO_ROOT}/.venv/bin/python"

"$PYTHON" - <<'PY' || warn "Wake model download failed; 'jarvis wake' will retry on first run"
import openwakeword.utils as utils
# Pulls the shared melspectrogram and embedding models plus the pretrained
# wake words, including hey_jarvis_v0.1.
utils.download_models()
print("  wake models ready")
PY

# ----------------------------------------------------------- piper binary
if command -v piper >/dev/null 2>&1; then
    say "Piper already on PATH at $(command -v piper)"
else
    say "Installing the Piper binary"
    ARCHIVE="piper_linux_x86_64.tar.gz"
    URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/${ARCHIVE}"
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT

    if curl -fsSL "$URL" -o "${TMP}/${ARCHIVE}"; then
        tar -xzf "${TMP}/${ARCHIVE}" -C "$TMP"
        rm -rf "${DATA_DIR}/piper-bin"
        mv "${TMP}/piper" "${DATA_DIR}/piper-bin"
        ln -sf "${DATA_DIR}/piper-bin/piper" "${BIN_DIR}/piper"
        echo "  installed to ${BIN_DIR}/piper"
        case ":${PATH}:" in
            *":${BIN_DIR}:"*) ;;
            *) warn "${BIN_DIR} is not on your PATH; add it to ~/.bashrc" ;;
        esac
    else
        warn "Could not download Piper. Jarvis will fall back to espeak-ng."
        warn "You can also try: pip install piper-tts"
    fi
fi

# ------------------------------------------------------------ piper voice
say "Downloading the Piper voice: ${VOICE}"
# Voice names encode their path: en_GB-alan-medium -> en/en_GB/alan/medium
LANG_FULL="${VOICE%%-*}"                 # en_GB
LANG_SHORT="${LANG_FULL%%_*}"            # en
REST="${VOICE#*-}"                       # alan-medium
SPEAKER="${REST%%-*}"                    # alan
QUALITY="${REST#*-}"                     # medium
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/${LANG_SHORT}/${LANG_FULL}/${SPEAKER}/${QUALITY}"

for suffix in onnx onnx.json; do
    target="${PIPER_DIR}/${VOICE}.${suffix}"
    if [[ -s "$target" ]]; then
        echo "  ${VOICE}.${suffix} already present"
        continue
    fi
    if curl -fsSL "${BASE}/${VOICE}.${suffix}" -o "$target"; then
        echo "  fetched ${VOICE}.${suffix}"
    else
        rm -f "$target"
        warn "Could not fetch ${VOICE}.${suffix} from ${BASE}"
        warn "Browse voices at https://huggingface.co/rhasspy/piper-voices"
    fi
done

# ---------------------------------------------------------- whisper model
say "Pre-downloading the Whisper model"
MODEL="${JARVIS_STT_MODEL:-base.en}"
"$PYTHON" - "$MODEL" <<'PY' || warn "Whisper download failed; it will retry on first use"
import sys
from faster_whisper import WhisperModel
model_name = sys.argv[1]
WhisperModel(model_name, device="cpu", compute_type="int8")
print(f"  whisper {model_name} cached")
PY

say "Models ready"
