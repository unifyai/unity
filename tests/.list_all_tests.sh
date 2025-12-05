#!/usr/bin/env bash
set -euo pipefail

# List test sessions across ALL terminals
#
# Usage:
#   ./.list_all_tests.sh         # List all sessions
#   ./.list_all_tests.sh --count # Show summary counts only

COUNT_ONLY=0

while (( "$#" )); do
  case "$1" in
    --count)
      COUNT_ONLY=1
      shift
      ;;
    -h|--help)
      echo "Usage: ./.list_all_tests.sh [--count]"
      echo ""
      echo "List test sessions across ALL terminals."
      echo ""
      echo "Options:"
      echo "  --count    Show summary counts only (no session names)"
      echo "  -h, --help Show this help"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

total_pending=0
total_passed=0
total_failed=0

for sock in /tmp/tmux-"$(id -u)"/unity*; do
  [ -e "$sock" ] || continue
  name=$(basename "$sock")

  # Get session list
  sessions=$(tmux -L "$name" ls 2>/dev/null || true)

  if [[ -z "$sessions" ]]; then
    continue
  fi

  # Count by status
  pending=$(echo "$sessions" | grep -c '^?' || true)
  passed=$(echo "$sessions" | grep -c '^o' || true)
  failed=$(echo "$sessions" | grep -c '^x' || true)
  count=$((pending + passed + failed))

  total_pending=$((total_pending + pending))
  total_passed=$((total_passed + passed))
  total_failed=$((total_failed + failed))

  if (( COUNT_ONLY )); then
    echo "$name: $count total ($pending ⏳, $passed ✅, $failed ❌)"
  else
    echo "=== $name ($count sessions) ==="
    echo "$sessions"
    echo
  fi
done

if (( total_pending + total_passed + total_failed == 0 )); then
  echo "No unity test sockets found."
  exit 0
fi

echo "──────────────────────────────────────"
echo "Total: $((total_pending + total_passed + total_failed)) sessions ($total_pending ⏳, $total_passed ✅, $total_failed ❌)"
