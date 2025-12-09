#!/usr/bin/env bash
set -euo pipefail

# Kill all failed tmux sessions (those starting with "f")
#
# Usage:
#   kill_failed.sh                    # Kill failed sessions in THIS terminal
#   kill_failed.sh -n                 # Dry run - show what would be killed
#   kill_failed.sh --all              # Kill failed sessions across ALL terminals
#   kill_failed.sh --socket <name>    # Kill failed sessions in a specific socket

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_shell_common.sh"

TMUX_SOCKET="$UNITY_TMUX_SOCKET"

# Wrapper for tmux commands
tmux_cmd() {
  tmux -L "$TMUX_SOCKET" "$@"
}

DRY_RUN=0
KILL_ALL=0
EXPLICIT_SOCKET=""

while (( "$#" )); do
  case "$1" in
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
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
      echo "Usage: kill_failed.sh [-n|--dry-run] [--all] [--socket <name>]"
      echo ""
      echo "Kill all failed tmux sessions (those starting with 'f')."
      echo ""
      echo "By default, only kills sessions from THIS terminal (isolated socket)."
      echo ""
      echo "Options:"
      echo "  -n, --dry-run      Show which sessions would be killed without killing them"
      echo "  --all              Kill failed sessions across ALL terminals"
      echo "  -s, --socket NAME  Kill failed sessions in a specific socket"
      echo "  -h, --help         Show this help"
      echo ""
      echo "Examples:"
      echo "  kill_failed.sh                              # Current terminal"
      echo "  kill_failed.sh --socket unity_dev_ttys042   # Specific socket"
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

# Collect all sockets to check
if (( KILL_ALL )); then
  # Find all unity* sockets
  SOCKETS=()
  while IFS= read -r sock; do
    [[ -n "$sock" ]] && SOCKETS+=( "$sock" )
  done < <(_get_unity_sockets)
  if (( ${#SOCKETS[@]} == 0 )); then
    echo "No unity test sockets found."
    exit 0
  fi
else
  SOCKETS=( "$TMUX_SOCKET" )
fi

# Get all session names starting with "f" (failed sessions)
# Uses timeout to avoid hanging on dead sockets
failed_sessions=()
for socket in "${SOCKETS[@]}"; do
  while IFS= read -r line; do
    session_name="${line%%:*}"
    if [[ "$session_name" == "f"* ]]; then
      failed_sessions+=( "$socket:$session_name" )
    fi
  done < <(LC_ALL=en_US.UTF-8 _tmux_ls "$socket")
done

if (( ${#failed_sessions[@]} == 0 )); then
  if (( KILL_ALL )); then
    echo "No failed sessions found across any terminal."
  else
    echo "No failed sessions found (socket: $TMUX_SOCKET)."
  fi
  exit 0
fi

echo "Found ${#failed_sessions[@]} failed session(s):"
for entry in "${failed_sessions[@]}"; do
  socket="${entry%%:*}"
  session="${entry#*:}"
  echo "  - [$socket] $session"
done

if (( DRY_RUN )); then
  echo ""
  echo "Dry run - no sessions killed."
  exit 0
fi

echo ""
for entry in "${failed_sessions[@]}"; do
  socket="${entry%%:*}"
  session="${entry#*:}"
  tmux -L "$socket" kill-session -t "$session" 2>/dev/null && echo "Killed: $session" || echo "Failed to kill: $session"
done

echo "Done."
