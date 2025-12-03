#!/usr/bin/env bash
set -euo pipefail

# Kill all failed tmux sessions (those starting with "x")
#
# Usage:
#   ./.kill_failed.sh        # Kill all failed sessions
#   ./.kill_failed.sh -n     # Dry run - show what would be killed

DRY_RUN=0

while (( "$#" )); do
  case "$1" in
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      echo "Usage: ./.kill_failed.sh [-n|--dry-run]"
      echo ""
      echo "Kill all failed tmux sessions (those starting with 'x')."
      echo ""
      echo "Options:"
      echo "  -n, --dry-run  Show which sessions would be killed without killing them"
      echo "  -h, --help     Show this help"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# Get all session names starting with "x" (failed sessions)
failed_sessions=()
while IFS= read -r line; do
  session_name="${line%%:*}"
  if [[ "$session_name" == x* ]]; then
    failed_sessions+=( "$session_name" )
  fi
done < <(tmux ls 2>/dev/null || true)

if (( ${#failed_sessions[@]} == 0 )); then
  echo "No failed sessions found."
  exit 0
fi

echo "Found ${#failed_sessions[@]} failed session(s):"
for s in "${failed_sessions[@]}"; do
  echo "  - $s"
done

if (( DRY_RUN )); then
  echo ""
  echo "Dry run - no sessions killed."
  exit 0
fi

echo ""
for s in "${failed_sessions[@]}"; do
  tmux kill-session -t "$s" 2>/dev/null && echo "Killed: $s" || echo "Failed to kill: $s"
done

echo "Done."
