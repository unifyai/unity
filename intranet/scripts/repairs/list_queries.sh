#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# list_queries.sh - List all active query run sockets and their sessions
#
# Similar to tests/list_runs.sh but for repairs query sockets.
#
# Usage:
#   list_queries.sh           # List sockets with active sessions
#   list_queries.sh --all     # Include empty sockets too
#   list_queries.sh --quiet   # Just list socket names (for scripting)
# ==============================================================================

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_repairs_common.sh"

CURRENT_SOCKET="$REPAIRS_TMUX_SOCKET"

QUIET=0
SHOW_ALL=0

while (( "$#" )); do
  case "$1" in
    -q|--quiet)
      QUIET=1
      shift
      ;;
    -a|--all)
      SHOW_ALL=1
      shift
      ;;
    -h|--help)
      cat << EOF
Usage: list_queries.sh [-q|--quiet] [-a|--all]

List all active query run sockets and their sessions.

By default, only shows sockets with active sessions.

Options:
  -a, --all    Include empty sockets (no active sessions)
  -q, --quiet  Just list socket names (for scripting)
  -h, --help   Show this help

Use the socket name with other commands:
  tmux -L <socket> ls                      List sessions in socket
  tmux -L <socket> attach -t <session>     Attach to a session
  tmux -L <socket> kill-session -t <name>  Kill a specific session
  tmux -L <socket> kill-server             Kill the entire server

Examples:
  # List all active query runs
  ./list_queries.sh

  # Get just socket names for scripting
  ./list_queries.sh --quiet

  # Attach to a session
  tmux -L \$(./list_queries.sh -q | head -1) attach
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# Find all repairs sockets
SOCKETS=()
while IFS= read -r sock; do
  [[ -n "$sock" ]] && SOCKETS+=( "$sock" )
done < <(_get_repairs_sockets)

if (( ${#SOCKETS[@]} == 0 )); then
  if (( QUIET )); then
    exit 0
  else
    echo "No query sockets found."
    echo ""
    echo "Start queries with:"
    echo "  ./parallel_queries.sh --all"
    exit 0
  fi
fi

# Count sockets with sessions (first pass)
sockets_with_sessions=0
empty_sockets=0
for socket in "${SOCKETS[@]}"; do
  sessions_output=$(_tmux_ls "$socket")
  if [[ -n "$sessions_output" ]]; then
    ((sockets_with_sessions++)) || true
  else
    ((empty_sockets++)) || true
  fi
done

if (( sockets_with_sessions == 0 )); then
  if (( QUIET )); then
    exit 0
  else
    if (( SHOW_ALL )); then
      echo "Found ${#SOCKETS[@]} socket(s), but none have active sessions."
    else
      echo "No active query runs found."
      echo ""
      echo "Tip: Use --all to see ${#SOCKETS[@]} empty socket(s)"
    fi
    exit 0
  fi
fi

# Detailed output (or quiet mode)
for socket in "${SOCKETS[@]}"; do
  sessions_output=$(_tmux_ls "$socket")
  has_sessions=0
  [[ -n "$sessions_output" ]] && has_sessions=1

  # Skip empty sockets unless --all
  if (( ! SHOW_ALL && ! has_sessions )); then
    continue
  fi

  if (( QUIET )); then
    echo "$socket"
    continue
  fi

  # Check if this is the current terminal's socket
  if [[ "$socket" == "$CURRENT_SOCKET" ]]; then
    marker="${GREEN}(current terminal)${NC}"
  else
    marker="${YELLOW}(other terminal)${NC}"
  fi

  echo ""
  echo -e "═══════════════════════════════════════════════════════════════════"
  echo -e "  Socket: ${BOLD}$socket${NC} $marker"
  echo -e "═══════════════════════════════════════════════════════════════════"

  if (( ! has_sessions )); then
    echo "  (no sessions)"
    echo ""
    continue
  fi

  # Count sessions by status
  running=0
  passed=0
  failed=0

  echo ""
  echo "  Sessions:"
  echo "  ─────────────────────────────────────────────────────────────────"

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    session_name="${line%%:*}"
    case "$session_name" in
      "r ⏳ "*)
        ((running++)) || true
        base="${session_name#r ⏳ }"
        print_pending "$base"
        ;;
      "p ✅ "*)
        ((passed++)) || true
        base="${session_name#p ✅ }"
        print_pass "$base"
        ;;
      "f ❌ "*)
        ((failed++)) || true
        base="${session_name#f ❌ }"
        print_fail "$base"
        ;;
      *)
        # Unknown status
        echo "    ? $session_name"
        ;;
    esac
  done <<< "$sessions_output"

  # Summary line
  total=$((running + passed + failed))
  echo ""
  echo "  ─────────────────────────────────────────────────────────────────"
  echo -n "  Summary: "

  if (( running > 0 )); then
    echo -en "${YELLOW}${running} running${NC}"
    if (( passed > 0 || failed > 0 )); then echo -n ", "; fi
  fi

  if (( passed > 0 )); then
    echo -en "${GREEN}${passed} passed${NC}"
    if (( failed > 0 )); then echo -n ", "; fi
  fi

  if (( failed > 0 )); then
    echo -en "${RED}${failed} failed${NC}"
  fi

  echo " (${total} total)"

  # Progress bar if queries are running
  if (( total > 0 && running > 0 )); then
    local completed=$((passed + failed))
    local pct=$((completed * 100 / total))
    local bar_width=40
    local filled=$((pct * bar_width / 100))
    local empty=$((bar_width - filled))

    echo -n "  Progress: ["
    if (( filled > 0 )); then
      echo -en "${GREEN}"
      printf '█%.0s' $(seq 1 $filled)
      echo -en "${NC}"
    fi
    if (( empty > 0 )); then
      printf '░%.0s' $(seq 1 $empty)
    fi
    echo "] ${pct}%"
  fi

  echo ""
done

if (( QUIET )); then
  exit 0
fi

# Show count of hidden sockets if not --all
if (( ! SHOW_ALL && empty_sockets > 0 )); then
  echo "($empty_sockets empty socket(s) hidden - use --all to show)"
  echo ""
fi

echo "─────────────────────────────────────────────────────────────────────"
echo "Commands:"
echo "  • Watch queries:    ./watch_queries.sh"
echo "  • Attach:           tmux -L <socket> attach -t '<session>'"
echo "  • Kill server:      tmux -L <socket> kill-server"
echo "─────────────────────────────────────────────────────────────────────"
