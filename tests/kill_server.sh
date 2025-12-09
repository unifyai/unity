#!/usr/bin/env bash
set -euo pipefail

# Kill the tmux server for the current terminal session
#
# Usage:
#   kill_server.sh                   # Kill THIS terminal's tmux server
#   kill_server.sh --all             # Kill ALL unity test tmux servers
#   kill_server.sh --socket <name>   # Kill a specific socket's server

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_shell_common.sh"

TMUX_SOCKET="$UNITY_TMUX_SOCKET"

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
      echo "Sends SIGTERM to processes before killing tmux for graceful shutdown."
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

# Helper: gracefully kill processes in a tmux socket before killing the server
_graceful_kill_socket() {
  local sock="$1"

  # Get all pane PIDs from all sessions in this socket
  local pids
  if [[ -n "$UNITY_TIMEOUT_CMD" ]]; then
    pids=$($UNITY_TIMEOUT_CMD tmux -L "$sock" list-panes -a -F '#{pane_pid}' 2>/dev/null || true)
  else
    pids=$(tmux -L "$sock" list-panes -a -F '#{pane_pid}' 2>/dev/null || true)
  fi

  if [[ -n "$pids" ]]; then
    # Send SIGTERM to process groups for graceful shutdown
    for pid in $pids; do
      if [[ -n "$pid" ]]; then
        # Kill process group to catch all child processes
        kill -TERM "-$pid" 2>/dev/null || true
      fi
    done
    # Brief wait for graceful shutdown
    sleep 0.2
  fi

  # Now kill the tmux server
  if [[ -n "$UNITY_TIMEOUT_CMD" ]]; then
    $UNITY_TIMEOUT_CMD tmux -L "$sock" kill-server 2>/dev/null
  else
    tmux -L "$sock" kill-server 2>/dev/null
  fi

  # Remove the socket file to prevent orphaned sockets
  rm -f "/tmp/tmux-$(id -u)/$sock" 2>/dev/null || true
}

if (( KILL_ALL )); then
  # Kill all unity* servers
  count=0
  while IFS= read -r name; do
    [[ -z "$name" ]] && continue
    if _graceful_kill_socket "$name"; then
      echo "Killed server: $name"
      ((count++)) || true
    fi
  done < <(_get_unity_sockets)
  if (( count == 0 )); then
    echo "No unity test servers found."
  else
    echo "Killed $count server(s)."
  fi
else
  # Kill just the specified (or current terminal's) server
  if _graceful_kill_socket "$TMUX_SOCKET"; then
    echo "Killed server: $TMUX_SOCKET"
  else
    echo "No server running for socket: $TMUX_SOCKET"
  fi
fi
