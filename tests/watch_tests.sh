#!/usr/bin/env bash
set -euo pipefail

# Watch test progress for the current terminal session
#
# Usage:
#   watch_tests.sh                      # Watch tests from THIS terminal
#   watch_tests.sh --all                # Watch tests from ALL terminals
#   watch_tests.sh --socket <name>      # Watch a specific socket (from any terminal)

# ---- Terminal-based isolation ----
# Uses the same socket detection as parallel_run.sh
_derive_socket_name() {
  local tty_id
  tty_id=$(tty 2>/dev/null)
  if [[ "$tty_id" == "not a tty" || -z "$tty_id" || ! "$tty_id" =~ ^/ ]]; then
    tty_id="pid$$"
  else
    tty_id=$(echo "$tty_id" | sed 's|/|_|g')
  fi
  echo "unity${tty_id}"
}

TMUX_SOCKET="${UNITY_TEST_SOCKET:-$(_derive_socket_name)}"

WATCH_ALL=0
EXPLICIT_SOCKET=""

while (( "$#" )); do
  case "$1" in
    --all)
      WATCH_ALL=1
      shift
      ;;
    -s|--socket)
      if [[ -n "${2-}" ]]; then
        EXPLICIT_SOCKET="$2"
        shift 2
      else
        echo "Error: --socket requires a socket name argument." >&2
        echo "Use 'list_runs.sh' to see available sockets." >&2
        exit 2
      fi
      ;;
    -h|--help)
      echo "Usage: watch_tests.sh [--all] [--socket <name>]"
      echo ""
      echo "Watch test progress in real-time."
      echo ""
      echo "By default, shows only tests from THIS terminal (isolated socket)."
      echo ""
      echo "Options:"
      echo "  --all              Show tests from ALL terminals"
      echo "  -s, --socket NAME  Watch a specific socket (use 'list_runs.sh' to find names)"
      echo "  -h, --help         Show this help"
      echo ""
      echo "Examples:"
      echo "  watch_tests.sh                           # Watch current terminal"
      echo "  watch_tests.sh --socket unity_dev_ttys042  # Watch orphaned socket"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# Use explicit socket if provided
if [[ -n "$EXPLICIT_SOCKET" ]]; then
  TMUX_SOCKET="$EXPLICIT_SOCKET"
fi

if (( WATCH_ALL )); then
  # Watch all unity sockets that have active sessions
  # Determine timeout command (needed to avoid hanging on dead sockets)
  if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD="timeout 1"
  elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD="gtimeout 1"
  else
    TIMEOUT_CMD=""
    echo "Warning: 'timeout' command not found. Dead sockets may cause slow refreshes." >&2
    echo "Install coreutils: brew install coreutils (macOS) or apt install coreutils (Linux)" >&2
    sleep 2
  fi

  # Build the watch command with timeout baked in
  # Note: We inline the timeout command to avoid complex quoting
  watch_script="
    found=0
    for sock in /tmp/tmux-\$(id -u)/unity*; do
      [ -e \"\$sock\" ] || continue
      name=\$(basename \"\$sock\")
      sessions=\$(${TIMEOUT_CMD} tmux -L \"\$name\" ls 2>/dev/null)
      [ -z \"\$sessions\" ] && continue
      found=1
      echo \"=== \$name ===\"
      echo \"\$sessions\"
      echo
    done
    if [ \"\$found\" -eq 0 ]; then
      echo \"(no active test sessions)\"
    fi
  "
  exec watch -n 0.5 "$watch_script"
else
  # Watch just the specified (or current terminal's) socket
  exec watch -n 0.5 "tmux -L $TMUX_SOCKET ls 2>/dev/null || echo '(no sessions)'"
fi
