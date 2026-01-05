#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# watch_queries.sh - Watch running repairs query sessions in tmux
#
# Similar to tests/watch_tests.sh but for repairs queries.
#
# Usage:
#   ./watch_queries.sh           # Watch all sessions
#   ./watch_queries.sh --summary # Just show status summary
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_repairs_common.sh"

TMUX_SOCKET="$REPAIRS_TMUX_SOCKET"

# Argument parsing
SUMMARY_ONLY=0
WATCH_INTERVAL=2

while (( "$#" )); do
  case "$1" in
    --summary|-s)
      SUMMARY_ONLY=1
      shift
      ;;
    --interval|-i)
      if [[ -n "${2-}" && "$2" =~ ^[0-9]+$ ]]; then
        WATCH_INTERVAL="$2"
        shift 2
      else
        error "--interval requires a number"
        exit 2
      fi
      ;;
    -h|--help)
      cat << EOF
Usage: watch_queries.sh [options]

Watch running repairs query sessions.

Options:
  -s, --summary       Show summary only (no continuous watch)
  -i, --interval N    Update interval in seconds (default: 2)
  -h, --help          Show this help

Examples:
  ./watch_queries.sh              # Continuous watch
  ./watch_queries.sh --summary    # One-time summary
  ./watch_queries.sh -i 5         # Update every 5 seconds
EOF
      exit 0
      ;;
    *)
      error "Unknown option: $1"
      exit 2
      ;;
  esac
done

# Check if tmux server is running
if ! tmux -L "$TMUX_SOCKET" ls >/dev/null 2>&1; then
  warn "No tmux server running on socket: $TMUX_SOCKET"
  echo ""
  echo "Start queries first with:"
  echo "  ./parallel_queries.sh --all"
  exit 0
fi

show_status() {
  clear

  echo ""
  echo "╔════════════════════════════════════════════════════════════════════╗"
  echo "║         Midland Heart Repairs - Query Status Monitor               ║"
  echo "╚════════════════════════════════════════════════════════════════════╝"
  echo ""
  echo "Socket: $TMUX_SOCKET"
  echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
  echo ""

  local pending=0
  local passed=0
  local failed=0

  echo "Sessions:"
  echo "─────────────────────────────────────────────────────────────────────"

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue

    session_name=$(echo "$line" | cut -d: -f1)

    case "$session_name" in
      "r ⏳ "*)
        ((pending++))
        base="${session_name#r ⏳ }"
        print_pending "$base"
        ;;
      "p ✅ "*)
        ((passed++))
        base="${session_name#p ✅ }"
        print_pass "$base"
        ;;
      "f ❌ "*)
        ((failed++))
        base="${session_name#f ❌ }"
        print_fail "$base"
        ;;
      *)
        # Unprefixed session (shouldn't happen normally)
        ((pending++))
        echo -e "  ${DIM}? $session_name${NC}"
        ;;
    esac
  done < <(tmux -L "$TMUX_SOCKET" ls 2>/dev/null || true)

  echo "─────────────────────────────────────────────────────────────────────"
  echo ""

  # Summary
  local total=$((pending + passed + failed))
  echo -n "Summary: "

  if (( passed > 0 )); then
    echo -en "${GREEN}${passed} passed${NC}"
    if (( pending > 0 || failed > 0 )); then echo -n ", "; fi
  fi

  if (( failed > 0 )); then
    echo -en "${RED}${failed} failed${NC}"
    if (( pending > 0 )); then echo -n ", "; fi
  fi

  if (( pending > 0 )); then
    echo -en "${YELLOW}${pending} running${NC}"
  fi

  echo " (${total} total)"
  echo ""

  # Progress bar
  if (( total > 0 )); then
    local completed=$((passed + failed))
    local pct=$((completed * 100 / total))
    local bar_width=50
    local filled=$((pct * bar_width / 100))
    local empty=$((bar_width - filled))

    echo -n "Progress: ["
    if (( filled > 0 )); then
      echo -en "${GREEN}"
      printf '█%.0s' $(seq 1 $filled)
      echo -en "${NC}"
    fi
    if (( empty > 0 )); then
      printf '░%.0s' $(seq 1 $empty)
    fi
    echo "] ${pct}%"
    echo ""
  fi

  if (( pending == 0 )); then
    echo "─────────────────────────────────────────────────────────────────────"
    if (( failed > 0 )); then
      echo -e "${RED}${BOLD}❌ Completed with ${failed} failure(s)${NC}"
    else
      echo -e "${GREEN}${BOLD}✅ All queries completed successfully!${NC}"
    fi
    echo "─────────────────────────────────────────────────────────────────────"
  fi

  echo ""
  echo "Commands:"
  echo "  • Attach:     tmux -L $TMUX_SOCKET attach -t '<session>'"
  echo "  • Kill all:   tmux -L $TMUX_SOCKET kill-server"
  echo "  • Exit watch: Ctrl+C"

  return $pending
}

if (( SUMMARY_ONLY )); then
  show_status
  exit 0
fi

# Continuous watch mode
trap 'echo ""; echo "Watch stopped."; exit 0' INT TERM

while true; do
  show_status
  pending=$?

  # If no pending queries, exit after showing final status
  if (( pending == 0 )); then
    echo ""
    echo "All queries finished. Exiting watch."
    exit 0
  fi

  sleep "$WATCH_INTERVAL"
done
