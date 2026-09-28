#!/usr/bin/env bash
#
# Install FlyBrain as a systemd *user* service, so it starts at login, restarts if it dies,
# and is stopped by name instead of by hunting for a process id:
#
#   deploy/install-service.sh              # install + enable, leave it stopped
#   deploy/install-service.sh --now        # install + enable + start (and restart if running)
#   deploy/install-service.sh --uninstall  # disable, stop, remove the unit
#
# It is idempotent: re-run it after `git pull` to pick up a changed unit. It never touches
# `.env`, and it never needs sudo for the service itself — only `loginctl enable-linger` does.
#
# The full picture, traps included, is docs/service.md.

set -euo pipefail

UNIT_NAME="flybrain.service"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TEMPLATE="$SCRIPT_DIR/$UNIT_NAME"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TARGET="$USER_UNIT_DIR/$UNIT_NAME"

say()  { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# Read one key from `.env` with the same rules flybrain/env.py uses: the last assignment
# wins, no inline-comment stripping (the app does not strip them either, and agreeing with
# the app matters more than being clever), and one layer of matching quotes is removed.
env_value() {
    local key="$1" line value
    [ -f "$ROOT/.env" ] || return 1
    line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}=" "$ROOT/.env" | tail -n 1 || true)"
    [ -n "$line" ] || return 1
    value="${line#*=}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    case "$value" in
        \"*\"|\'*\') value="${value:1:${#value}-2}" ;;
    esac
    printf '%s' "$value"
}

is_true() {
    case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

start_now=0
uninstall=0
for arg in "$@"; do
    case "$arg" in
        --now) start_now=1 ;;
        --uninstall|--remove) uninstall=1 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $arg" ;;
    esac
done

# ---------------------------------------------------------------- preconditions

command -v systemctl >/dev/null 2>&1 || die "systemctl not found; this installer is for systemd"
systemctl --user show-environment >/dev/null 2>&1 \
    || die "no reachable systemd user manager (no D-Bus session). Run this from a normal login, or enable linger first."
[ -f "$TEMPLATE" ] || die "unit template not found: $TEMPLATE"

# ------------------------------------------------------------------- uninstall

if [ "$uninstall" = 1 ]; then
    systemctl --user disable --now "$UNIT_NAME" 2>/dev/null || true
    rm -f "$TARGET"
    systemctl --user daemon-reload
    systemctl --user reset-failed "$UNIT_NAME" 2>/dev/null || true
    say "removed $TARGET"
    say "FlyBrain's own files (.env, data/recordings/, data/experiments/) were left alone."
    exit 0
fi

# --------------------------------------------------------------------- install

[ -x "$ROOT/.venv/bin/python" ] \
    || die "no .venv/bin/python under $ROOT — run 'uv sync' first (see README.md)"

mkdir -p "$USER_UNIT_DIR"
if grep -q '%h/FlyBrain' "$TEMPLATE"; then
    # The shipped unit defaults to the README's clone path; rewrite it to this checkout so a
    # non-standard clone path does not silently run the wrong interpreter (or nothing at all).
    sed "s|%h/FlyBrain|$ROOT|g" "$TEMPLATE" > "$TARGET.tmp"
else
    cp "$TEMPLATE" "$TARGET.tmp"
fi
mv "$TARGET.tmp" "$TARGET"

systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME" >/dev/null

if [ "$start_now" = 1 ]; then
    # `restart` rather than `start`: re-running the installer should pick up the new unit and
    # a fresh interpreter, not report "already active" and leave the old process in place.
    systemctl --user restart "$UNIT_NAME"
fi

# ------------------------------------------------------------------- post-flight

if [ "$(loginctl show-user "${USER:-$(id -un)}" -p Linger --value 2>/dev/null || echo unknown)" != "yes" ]; then
    warn "linger is off: the service starts when you log in and stops when your last session ends."
    warn "for a house that runs while you are logged out, enable it once with:"
    warn "    sudo loginctl enable-linger ${USER:-$(id -un)}"
fi

if [ ! -f "$ROOT/.env" ]; then
    warn "no .env: the service will run the simulated home, which is fine for watching and"
    warn "pointless for controlling. Copy .env.example to .env when you want the real one."
else
    if ! is_true "$(env_value FLYBRAIN_ALWAYS_ON || true)"; then
        warn "FLYBRAIN_ALWAYS_ON is not enabled in .env. The service will serve the dashboard,"
        warn "but the brain will only step while a browser is connected. Set FLYBRAIN_ALWAYS_ON=1"
        warn "for a control loop that runs unattended."
    fi
    mode="$(env_value HA_MODE || true)"
    if [ -n "$mode" ] && [ "$mode" != "rest" ]; then
        warn "HA_MODE=$mode: the loop is talking to the simulated home, not yours."
    fi
    if [ "$mode" = "rest" ] && ! is_true "$(env_value HA_DRY_RUN || true)"; then
        warn "HA_DRY_RUN is off and HA_MODE=rest: the loop may now write to a real light."
    fi
fi

say ""
say "installed $TARGET"
if [ "$start_now" != 1 ]; then
    say "start it with:      systemctl --user start $UNIT_NAME"
fi
say "follow the logs:    journalctl --user -u $UNIT_NAME -f"
say "check it:           systemctl --user status $UNIT_NAME"
say "reload after edits: systemctl --user daemon-reload && systemctl --user restart $UNIT_NAME"
say "dashboard:          http://127.0.0.1:8765/"
say "docs:               $ROOT/docs/service.md"
