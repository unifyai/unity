#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# kill_failed_queries.sh - Kill all failed query sessions (those starting with "f")
#
# Similar to tests/kill_failed.sh but for repairs query sockets.
#
# Usage:
#   kill_failed_queries.sh                    # Kill failed sessions in THIS terminal
#   kill_failed_queries.sh -n                 # Dry run - show what would be killed
#   kill_failed_queries.sh --all              # Kill failed sessions across ALL terminals
#   kill_failed_queries.sh --socket <name>    # Kill failed sessions in a specific socket
# ==============================================================================

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_repairs_common.sh"

TMUX_SOCKET="$REPAIRS_TMUX_SOCKET"

# Wrapper for tmux commands (uses locale from _repairs_common.sh)
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
        error "--socket requires a socket name argument."
        echo "Use './list_queries.sh' to see available sockets." >&2
        exit 2
      fi
      ;;
    -h|--help)
      cat << EOF
Usage: kill_failed_queries.sh [-n|--dry-run] [--all] [--socket <name>]

Kill all failed query sessions (those starting with 'f ❌').

By default, only kills sessions from THIS terminal (isolated socket).

Options:
  -n, --dry-run      Show which sessions would be killed without killing them
  --all              Kill failed sessions across ALL terminals
  -s, --socket NAME  Kill failed sessions in a specific socket
  -h, --help         Show this help

Examples:
  ./kill_failed_queries.sh                              # Current terminal
  ./kill_failed_queries.sh --dry-run                    # Preview what would be killed
  ./kill_failed_queries.sh --all                        # All terminals
  ./kill_failed_queries.sh --socket repairs_dev_pts_0   # Specific socket
EOF
      exit 0
      ;;
    *)
      error "Unknown argument: $1"
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
  # Find all repairs* sockets
  SOCKETS=()
  while IFS= read -r sock; do
    [[ -n "$sock" ]] && SOCKETS+=( "$sock" )
  done < <(_get_repairs_sockets)
  if (( ${#SOCKETS[@]} == 0 )); then
    info "No repairs query sockets found."
    exit 0
  fi
else
  SOCKETS=( "$TMUX_SOCKET" )
fi

# Get all session names starting with "f" (failed sessions)
failed_sessions=()
for socket in "${SOCKETS[@]}"; do
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    session_name="${line%%:*}"
    if [[ "$session_name" == "f"* ]]; then
      failed_sessions+=( "$socket:$session_name" )
    fi
  done < <(_tmux_ls "$socket")
done

if (( ${#failed_sessions[@]} == 0 )); then
  if (( KILL_ALL )); then
    success "No failed sessions found across any terminal."
  else
    success "No failed sessions found (socket: $TMUX_SOCKET)."
  fi
  exit 0
fi

echo ""
echo -e "${RED}${BOLD}Found ${#failed_sessions[@]} failed session(s):${NC}"
echo ""
for entry in "${failed_sessions[@]}"; do
  socket="${entry%%:*}"
  session="${entry#*:}"
  base="${session#f ❌ }"
  echo -e "  ${RED}f ❌${NC} [$socket] $base"
done

if (( DRY_RUN )); then
  echo ""
  warn "Dry run - no sessions killed."
  echo ""
  echo "To kill these sessions, run without --dry-run"
  exit 0
fi

echo ""
echo "Killing failed sessions..."
echo ""

killed=0
for entry in "${failed_sessions[@]}"; do
  socket="${entry%%:*}"
  session="${entry#*:}"
  base="${session#f ❌ }"
  if tmux -L "$socket" kill-session -t "$session" 2>/dev/null; then
    print_pass "Killed: $base"
    ((killed++)) || true
  else
    print_fail "Failed to kill: $base"
  fi
done

echo ""
success "Killed $killed of ${#failed_sessions[@]} failed session(s)."
