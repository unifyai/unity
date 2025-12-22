#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# kill_server_queries.sh - Kill the tmux server for query sessions
#
# Similar to tests/kill_server.sh but for repairs query sockets.
#
# Usage:
#   kill_server_queries.sh                   # Kill THIS terminal's tmux server
#   kill_server_queries.sh --all             # Kill ALL repairs* tmux servers
#   kill_server_queries.sh --socket <name>   # Kill a specific socket's server
# ==============================================================================

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SCRIPT_DIR/_repairs_common.sh"

TMUX_SOCKET="$REPAIRS_TMUX_SOCKET"

KILL_ALL=0
EXPLICIT_SOCKET=""
FORCE=0

while (( "$#" )); do
  case "$1" in
    --all)
      KILL_ALL=1
      shift
      ;;
    -f|--force)
      FORCE=1
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
Usage: kill_server_queries.sh [--all] [--socket <name>] [-f|--force]

Kill the tmux server for query sessions and clean up processes.

By default, kills THIS terminal's tmux server (isolated socket).
Sends SIGTERM to processes before killing tmux for graceful shutdown.

Options:
  --all              Kill ALL repairs* tmux servers across all terminals
  -s, --socket NAME  Kill a specific socket's server
  -f, --force        Skip confirmation prompt
  -h, --help         Show this help

Examples:
  ./kill_server_queries.sh                              # Current terminal
  ./kill_server_queries.sh --all                        # All repairs servers
  ./kill_server_queries.sh --socket repairs_dev_pts_0   # Specific socket
  ./kill_server_queries.sh --all --force                # No confirmation
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

# Helper: gracefully kill processes in a tmux socket before killing the server
_graceful_kill_socket() {
  local sock="$1"

  # Get all pane PIDs from all sessions in this socket
  local pids
  if [[ -n "$REPAIRS_TIMEOUT_CMD" ]]; then
    pids=$($REPAIRS_TIMEOUT_CMD tmux -L "$sock" list-panes -a -F '#{pane_pid}' 2>/dev/null || true)
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
  if [[ -n "$REPAIRS_TIMEOUT_CMD" ]]; then
    $REPAIRS_TIMEOUT_CMD tmux -L "$sock" kill-server 2>/dev/null || true
  else
    tmux -L "$sock" kill-server 2>/dev/null || true
  fi

  # Remove the socket file to prevent orphaned sockets
  rm -f "/tmp/tmux-$(id -u)/$sock" 2>/dev/null || true
}

# Count sessions before killing
_count_sessions() {
  local sock="$1"
  local count=0
  while IFS= read -r line; do
    [[ -n "$line" ]] && ((count++)) || true
  done < <(_tmux_ls "$sock")
  echo "$count"
}

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           Midland Heart Repairs - Kill Query Server                ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

if (( KILL_ALL )); then
  # Find all repairs* sockets
  SOCKETS=()
  while IFS= read -r sock; do
    [[ -n "$sock" ]] && SOCKETS+=( "$sock" )
  done < <(_get_repairs_sockets)

  if (( ${#SOCKETS[@]} == 0 )); then
    info "No repairs query servers found."
    exit 0
  fi

  # Count total sessions
  total_sessions=0
  echo "Found ${#SOCKETS[@]} server(s):"
  for sock in "${SOCKETS[@]}"; do
    count=$(_count_sessions "$sock")
    total_sessions=$((total_sessions + count))
    if [[ "$sock" == "$REPAIRS_TMUX_SOCKET" ]]; then
      marker="${GREEN}(current)${NC}"
    else
      marker="${YELLOW}(other)${NC}"
    fi
    echo -e "  • $sock - $count session(s) $marker"
  done
  echo ""

  # Confirm unless forced
  if (( ! FORCE && total_sessions > 0 )); then
    echo -e "${YELLOW}This will kill $total_sessions session(s) across ${#SOCKETS[@]} server(s).${NC}"
    read -p "Continue? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      warn "Aborted."
      exit 0
    fi
    echo ""
  fi

  # Kill all servers
  killed=0
  for sock in "${SOCKETS[@]}"; do
    _graceful_kill_socket "$sock"
    success "Killed server: $sock"
    ((killed++)) || true
  done

  echo ""
  success "Killed $killed server(s)."

else
  # Kill just the specified (or current terminal's) server
  count=$(_count_sessions "$TMUX_SOCKET")

  if (( count == 0 )); then
    # Check if server exists at all
    if ! tmux -L "$TMUX_SOCKET" ls >/dev/null 2>&1; then
      info "No server running for socket: $TMUX_SOCKET"
      exit 0
    fi
  fi

  echo "Server: $TMUX_SOCKET"
  echo "Sessions: $count"
  echo ""

  # Confirm unless forced
  if (( ! FORCE && count > 0 )); then
    echo -e "${YELLOW}This will kill $count session(s).${NC}"
    read -p "Continue? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
      warn "Aborted."
      exit 0
    fi
    echo ""
  fi

  _graceful_kill_socket "$TMUX_SOCKET"
  success "Killed server: $TMUX_SOCKET"
fi

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "To start new queries:"
echo "  ./parallel_queries.sh --all"
echo "─────────────────────────────────────────────────────────────────────"
