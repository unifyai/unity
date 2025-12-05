#!/usr/bin/env bash
set -euo pipefail

# Kill the tmux server for the current terminal session
#
# Usage:
#   kill_server.sh                   # Kill THIS terminal's tmux server
#   kill_server.sh --all             # Kill ALL unity test tmux servers
#   kill_server.sh --socket <name>   # Kill a specific socket's server

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

KILL_ALL=0
EXPLICIT_SOCKET=""

while (( "$#" )); do
  case "$1" in
    --all)
      KILL_ALL=1
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
      echo "Usage: kill_server.sh [--all] [--socket <name>]"
      echo ""
      echo "Kill the tmux server for test sessions."
      echo ""
      echo "By default, kills only THIS terminal's tmux server (isolated socket)."
      echo ""
      echo "Options:"
      echo "  --all              Kill ALL unity test tmux servers across all terminals"
      echo "  -s, --socket NAME  Kill a specific socket's server"
      echo "  -h, --help         Show this help"
      echo ""
      echo "Examples:"
      echo "  kill_server.sh                              # Current terminal"
      echo "  kill_server.sh --socket unity_dev_ttys042   # Specific socket"
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

if (( KILL_ALL )); then
  # Kill all unity* servers
  count=0
  for sock in /tmp/tmux-"$(id -u)"/unity*; do
    [ -e "$sock" ] || continue
    name=$(basename "$sock")
    if tmux -L "$name" kill-server 2>/dev/null; then
      echo "Killed server: $name"
      ((count++)) || true
    fi
  done
  if (( count == 0 )); then
    echo "No unity test servers found."
  else
    echo "Killed $count server(s)."
  fi
else
  # Kill just the specified (or current terminal's) server
  if tmux -L "$TMUX_SOCKET" kill-server 2>/dev/null; then
    echo "Killed server: $TMUX_SOCKET"
  else
    echo "No server running for socket: $TMUX_SOCKET"
  fi
fi
