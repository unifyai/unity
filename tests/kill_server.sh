#!/usr/bin/env bash
set -euo pipefail

# Kill the tmux server for the current terminal session
#
# Usage:
#   kill_server.sh                   # Kill THIS terminal's tmux server
#   kill_server.sh --all             # Kill ALL unity* tmux servers
#   kill_server.sh --global          # Kill ALL tmux servers for this user
#   kill_server.sh --socket <name>   # Kill a specific socket's server
#   kill_server.sh --purge           # Kill ALL orphaned pytest processes from unity tests

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_shell_common.sh"

TMUX_SOCKET="$UNITY_TMUX_SOCKET"

KILL_ALL=0
KILL_GLOBAL=0
KILL_PURGE=0
EXPLICIT_SOCKET=""

while (( "$#" )); do
  case "$1" in
    --all)
      KILL_ALL=1
      shift
      ;;
    --global)
      KILL_GLOBAL=1
      shift
      ;;
    --purge)
      KILL_PURGE=1
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
      echo "Usage: kill_server.sh [--all] [--global] [--purge] [--socket <name>]"
      echo ""
      echo "Kill the tmux server for test sessions."
      echo ""
      echo "By default, kills only THIS terminal's tmux server (isolated socket)."
      echo "Sends SIGTERM to processes before killing tmux for graceful shutdown."
      echo ""
      echo "Options:"
      echo "  --all              Kill ALL unity* tmux servers across all terminals"
      echo "  --global           Kill ALL tmux servers for this user (any name)"
      echo "  --purge            Kill ALL orphaned pytest/python processes from unity tests"
      echo "                     (useful after crashes that leave processes behind)"
      echo "  -s, --socket NAME  Kill a specific socket's server"
      echo "  -h, --help         Show this help"
      echo ""
      echo "Examples:"
      echo "  kill_server.sh                              # Current terminal"
      echo "  kill_server.sh --socket unity_dev_ttys042   # Specific socket"
      echo "  kill_server.sh --all                        # All unity* servers"
      echo "  kill_server.sh --global                     # All tmux servers (any name)"
      echo "  kill_server.sh --purge                      # Kill orphaned test processes"
      echo "  kill_server.sh --global --purge             # Full cleanup: servers + orphans"
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

  # Now kill the tmux server (ignore errors if server doesn't exist)
  if [[ -n "$UNITY_TIMEOUT_CMD" ]]; then
    $UNITY_TIMEOUT_CMD tmux -L "$sock" kill-server 2>/dev/null || true
  else
    tmux -L "$sock" kill-server 2>/dev/null || true
  fi

  # Remove the socket file to prevent orphaned sockets
  rm -f "/tmp/tmux-$(id -u)/$sock" 2>/dev/null || true
}

if (( KILL_GLOBAL )); then
  # Kill all tmux servers for this user, regardless of name
  count=0
  for sock in /tmp/tmux-"$(id -u)"/*; do
    [[ -e "$sock" ]] || continue
    name=$(basename "$sock")
    _graceful_kill_socket "$name"
    echo "Killed server: $name"
    ((count++)) || true
  done
  if (( count == 0 )); then
    echo "No tmux servers found."
  else
    echo "Killed $count server(s)."
  fi
elif (( KILL_ALL )); then
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

# ---- Purge orphaned processes ----
# When tmux sessions are killed abruptly (e.g., socket deleted by race condition),
# pytest and bash processes can become orphaned. This finds and kills them.
if (( KILL_PURGE )); then
  echo ""
  echo "Purging orphaned test processes..."

  purge_count=0

  # Find ALL pytest processes running from unity's virtualenv
  # Pattern: unity/.venv.*pytest matches any pytest process from the unity project
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    kill -TERM "$pid" 2>/dev/null || true
    ((purge_count++)) || true
  done < <(pgrep -f "unity/.venv.*pytest" 2>/dev/null || true)

  # Find orphaned bash/tmux processes with unity test markers in command line
  # These have UNITY_TEST_SOCKET, unity_dev_, or unity_test_ in their args
  while IFS= read -r pid; do
    [[ -z "$pid" ]] && continue
    # Don't kill the current shell or its parent
    [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
    cmdline=$(ps -o command= -p "$pid" 2>/dev/null || true)
    # Only kill if it's a shell process, not this script itself
    if [[ "$cmdline" == "bash"* || "$cmdline" == "tmux"* ]]; then
      kill -TERM "$pid" 2>/dev/null || true
      ((purge_count++)) || true
    fi
  done < <(pgrep -f "UNITY_TEST_SOCKET|unity_dev_|unity_test_" 2>/dev/null || true)

  # Brief wait for SIGTERM to take effect
  if (( purge_count > 0 )); then
    sleep 1
    # Follow up with SIGKILL for any stubborn processes
    pkill -9 -f "unity/.venv.*pytest" 2>/dev/null || true
    echo "Terminated $purge_count orphaned process(es)."
  else
    echo "No orphaned test processes found."
  fi
fi
