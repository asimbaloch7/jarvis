#!/usr/bin/env bash
# Install the systemd --user unit so Jarvis starts on login.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_DIR="${HOME}/.config/systemd/user"
UNIT="${UNIT_DIR}/jarvis.service"
BIN="${REPO_ROOT}/.venv/bin/jarvis"

if [[ ! -x "$BIN" ]]; then
    echo "error: ${BIN} not found. Run ./scripts/setup.sh first." >&2
    exit 1
fi

mkdir -p "$UNIT_DIR"

# WorkingDirectory cannot be quoted (systemd then rejects the unit). It also
# does not word-split, so spaces in the path are fine unquoted. ExecStart
# does word-split, so the binary path must be quoted.
python3 - "$REPO_ROOT" "$UNIT" <<'PY'
import sys
from pathlib import Path

root, dest = sys.argv[1], Path(sys.argv[2])
template = Path(root) / "systemd" / "jarvis.service"
text = template.read_text().replace("__JARVIS_HOME__", root)
dest.write_text(text)
print(f"wrote {dest}")
PY

systemctl --user daemon-reload
systemctl --user enable jarvis.service
if ! systemctl --user start jarvis.service; then
    echo >&2
    echo "error: the service did not start. Details:" >&2
    systemctl --user --no-pager status jarvis.service || true
    echo >&2
    echo "If it says the microphone is busy, stop a manual 'jarvis run' first:" >&2
    echo "    pkill -f 'jarvis run' || true" >&2
    echo "    systemctl --user start jarvis" >&2
    exit 1
fi

echo
echo "Installed ${UNIT}"
systemctl --user --no-pager --lines=15 status jarvis.service || true
cat <<EOF

Jarvis is now a user service. It starts when you log in.

  systemctl --user status jarvis      # is it alive
  systemctl --user restart jarvis     # after changing code or config
  systemctl --user stop jarvis        # free the microphone
  systemctl --user disable --now jarvis
  journalctl --user -u jarvis -f      # live spoken-command log

Do not also run 'jarvis run' in a terminal while the service is up —
they would fight over the microphone.

Then say:  Hey Jarvis
Wait for the beep, then give a command, e.g.:  Start my dev environment.

EOF
