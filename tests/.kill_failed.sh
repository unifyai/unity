#!/usr/bin/env bash
set -euo pipefail

# Kill all failed tmux sessions (those starting with "x")
#
# Usage:
#   ./.kill_failed.sh        # Kill all failed sessions in THIS terminal
#   ./.kill_failed.sh -n     # Dry run - show what would be killed
#   ./.kill_failed.sh --all  # Kill failed sessions across ALL terminals

# ---- Terminal-based isolation ----
# Uses the same socket detection as .parallel_run.sh
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

# Wrapper for tmux commands
tmux_cmd() {
  tmux -L "$TMUX_SOCKET" "$@"
}

DRY_RUN=0
KILL_ALL=0

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
    -h|--help)
      echo "Usage: ./.kill_failed.sh [-n|--dry-run] [--all]"
      echo ""
      echo "Kill all failed tmux sessions (those starting with 'x')."
      echo ""
      echo "By default, only kills sessions from THIS terminal (isolated socket)."
      echo ""
      echo "Options:"
      echo "  -n, --dry-run  Show which sessions would be killed without killing them"
      echo "  --all          Kill failed sessions across ALL terminals"
      echo "  -h, --help     Show this help"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# Collect all sockets to check
if (( KILL_ALL )); then
  # Find all unity* sockets
  SOCKETS=()
  for sock in /tmp/tmux-"$(id -u)"/unity*; do
    [ -e "$sock" ] || continue
    SOCKETS+=( "$(basename "$sock")" )
  done
  if (( ${#SOCKETS[@]} == 0 )); then
    echo "No unity test sockets found."
    exit 0
  fi
else
  SOCKETS=( "$TMUX_SOCKET" )
fi

# Get all session names starting with "x" (failed sessions)
failed_sessions=()
for socket in "${SOCKETS[@]}"; do
  while IFS= read -r line; do
    session_name="${line%%:*}"
    if [[ "$session_name" == "x"* ]]; then
      failed_sessions+=( "$socket:$session_name" )
    fi
  done < <(LC_ALL=en_US.UTF-8 tmux -L "$socket" ls 2>/dev/null || true)
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
