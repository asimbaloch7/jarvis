#!/usr/bin/env bash
# Install the Python dependencies into .venv, one package at a time.
#
# Why not a single `pip install -e .`? Two reasons:
#
# 1. On an unreliable connection the resolver backtracks through candidate
#    versions of faster-whisper, refetches metadata for each one, and if a
#    single request drops it reports "no matching distribution: tokenizers"
#    even though the wheel is on PyPI. That is the error you just hit.
# 2. Installing one package per pip invocation keeps each step small, retries
#    it independently, and leaves finished wheels in pip's cache, so re-running
#    this script resumes instead of starting over.
#
# Safe to run as many times as it takes.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PIP="${REPO_ROOT}/.venv/bin/pip"
ATTEMPTS="${JARVIS_PIP_ATTEMPTS:-5}"
PIP_FLAGS=(--retries 10 --timeout 120)

# Native wheels first, while you are most likely still watching. tokenizers
# and ctranslate2 MUST be in before faster-whisper, otherwise pip starts
# hunting through older faster-whisper releases and the install dies.
#
# faster-whisper is installed with --no-deps because its requirements are
# the lines above it.
PACKAGES=(
    "numpy>=1.24,<3"
    "PyYAML>=6.0"
    "python-dotenv>=1.0"
    "sounddevice>=0.4.6"
    "onnxruntime>=1.17,<2"
    "tokenizers==0.22.1"
    "ctranslate2>=4.0,<5"
    "av>=11"
    "huggingface-hub>=0.21"
    "tqdm"
    "scipy>=1.11,<2"
    "scikit-learn>=1.3,<2"
    "requests>=2.28,<3"
    "google-genai>=1.0.0"
)

say()  { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()   { printf '\033[32m    ok: %s\033[0m\n' "$1"; }
warn() { printf '\033[33m    %s\033[0m\n' "$1"; }
fail() { printf '\033[31m    failed: %s\033[0m\n' "$1"; }

if [[ ! -x "$PIP" ]]; then
    echo "error: ${PIP} not found. Create the venv first:" >&2
    echo "    python3.12 -m venv .venv" >&2
    exit 1
fi

PY_VERSION="$("${REPO_ROOT}/.venv/bin/python" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
if [[ "$PY_VERSION" != "3.10" && "$PY_VERSION" != "3.11" && "$PY_VERSION" != "3.12" ]]; then
    echo "error: .venv is Python ${PY_VERSION}, but onnxruntime and ctranslate2" >&2
    echo "       need 3.10-3.12. Rebuild it:" >&2
    echo "    rm -rf .venv && python3.12 -m venv .venv" >&2
    exit 1
fi

install_one() {
    local extra_flags=()
    if [[ "$1" == "--no-deps" ]]; then
        extra_flags=(--no-deps)
        shift
    fi
    local spec="$1"
    local name="${spec%%[<>=]*}"
    local attempt

    for attempt in $(seq 1 "$ATTEMPTS"); do
        printf '    attempt %d/%d: %s\n' "$attempt" "$ATTEMPTS" "$spec"
        if "$PIP" install "${PIP_FLAGS[@]}" "${extra_flags[@]}" "$spec"; then
            ok "$name"
            return 0
        fi
        warn "attempt ${attempt} failed, retrying..."
        sleep 3
    done

    fail "$name after ${ATTEMPTS} attempts"
    return 1
}

say "Installing dependencies into .venv (Python ${PY_VERSION})"
echo "    Each package is retried up to ${ATTEMPTS} times."
echo "    Completed downloads are cached, so re-running resumes."

FAILED=()
for spec in "${PACKAGES[@]}"; do
    install_one "$spec" || FAILED+=("$spec")
done

# openwakeword's PyPI metadata still requires tflite-runtime on Linux. That
# package has no Python 3.12 wheel (last release targets <3.12). We already
# force inference_framework="onnx", so skip tflite and install the package
# itself with --no-deps. scipy/scikit-learn/requests/onnxruntime are above.
install_one --no-deps "openwakeword==0.6.0" || FAILED+=("openwakeword==0.6.0")

# faster-whisper last, with --no-deps: tokenizers/ctranslate2/onnxruntime/av
# are already installed above. Without --no-deps, pip re-enters the resolver
# and can invent the "tokenizers has no matching distribution" error again.
install_one --no-deps "faster-whisper==1.2.1" || FAILED+=("faster-whisper==1.2.1")

if (( ${#FAILED[@]} > 0 )); then
    say "Some packages did not install"
    for spec in "${FAILED[@]}"; do
        echo "    - ${spec}"
    done
    cat <<EOF

Re-run this script and it will skip packages that are already installed:

    ./scripts/install_python_deps.sh

If a package still fails, the error above it is the real cause — not a
dropped connection. The most common one is a native wheel that does not
exist for this Python version.

EOF
    exit 1
fi

# --no-deps because everything above is already resolved. This step only
# registers the project and creates the `jarvis` command.
say "Installing Jarvis itself"
if ! "$PIP" install "${PIP_FLAGS[@]}" --no-deps -e .; then
    fail "editable install of jarvis"
    exit 1
fi
ok "jarvis command installed"

# Optional: better endpointing in noisy rooms. Never fatal.
"$PIP" install "${PIP_FLAGS[@]}" --quiet webrtcvad-wheels >/dev/null 2>&1 \
    && ok "webrtcvad (optional)" \
    || warn "webrtcvad unavailable, the energy VAD will be used instead"

say "Dependencies installed"
cat <<'EOF'
    If the prompt already shows (.venv), just run:
        jarvis doctor

    Otherwise activate the venv first (source, do not execute):
        source .venv/bin/activate
        jarvis doctor

    Do not run `.venv/bin/activate` as a command — bash will say
    "Permission denied". The `source` is required.
EOF

