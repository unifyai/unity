#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Resource Monitor Dashboard
# ============================================================================
#
# Launches a tmux-based dashboard for monitoring system resources during
# test runs. Optimized for network I/O heavy workloads (LLM API calls, etc.)
#
# Usage:
#   ./monitor_resources.sh          # Launch the dashboard
#   ./monitor_resources.sh --help   # Show help
#
# Dashboard Layout:
#   ┌──────────────────────────────────────────────┐
#   │                    htop                      │
#   │          (CPU, Memory, Processes)            │
#   ├──────────────────────────────────────────────┤
#   │                   nettop                     │
#   │         (Per-process Network I/O)            │
#   ├──────────────────────┬───────────────────────┤
#   │   File Descriptors   │   TCP Connections     │
#   │  (Python processes)  │   (Active sockets)    │
#   └──────────────────────┴───────────────────────┘
#
# Requirements:
#   - tmux (brew install tmux)
#   - htop (brew install htop) - optional, falls back to top
#   - nettop (built into macOS)
#
# Exit:
#   Press Ctrl-C in any pane, or close the terminal
#   Or run: tmux kill-session -t unity-monitor

SESSION_NAME="unity-monitor"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
  cat << 'EOF'
Resource Monitor Dashboard
==========================

Launches a tmux-based dashboard for monitoring system resources during
test runs. Optimized for network I/O heavy workloads.

USAGE:
  ./monitor_resources.sh          Launch the dashboard
  ./monitor_resources.sh --help   Show this help

DASHBOARD PANES:

  1. htop (top half)
     - CPU usage per core
     - Memory usage (used/cached/buffers)
     - Process list sorted by CPU
     - Swap usage

  2. nettop (middle)
     - Per-process network I/O (bytes in/out)
     - Active network connections
     - Real-time bandwidth per process

  3. File Descriptors (bottom left)
     - Count of open file descriptors for Python processes
     - Useful for detecting connection leaks
     - Warning threshold: ulimit (usually 256-4096)

  4. TCP Connections (bottom right)
     - Count of ESTABLISHED TCP connections
     - Count of TIME_WAIT connections (slow to recycle)
     - Total listening sockets

INTERPRETING THE METRICS:

  When running parallel tests with heavy network I/O:

  CPU:
    - Expect moderate usage (20-50%) from async event loops
    - SSL/TLS handshakes are CPU-intensive
    - High CPU with low network = potential bottleneck

  Memory:
    - Watch for growth during long test runs
    - Response buffering can consume significant memory
    - Cached memory is fine (kernel will release if needed)

  Network:
    - nettop shows per-process bandwidth
    - Look for Python processes with high bytes/sec
    - Many ESTABLISHED connections = high parallelism

  File Descriptors:
    - Each TCP connection = 1 file descriptor
    - If count approaches ulimit, you'll see failures
    - Fix: increase with 'ulimit -n 4096' before running tests

  TIME_WAIT:
    - Normal after connections close (lasts ~60s on macOS)
    - Very high counts may indicate excessive connection churn
    - Consider connection pooling if problematic

TIPS:

  Increase file descriptor limit before heavy test runs:
    ulimit -n 4096

  Check your current limit:
    ulimit -n

  macOS network tuning (for extreme parallelism):
    sudo sysctl -w kern.maxfiles=65536
    sudo sysctl -w kern.maxfilesperproc=65536

KEYBOARD SHORTCUTS (inside tmux):

  Ctrl-b + arrow    Move between panes
  Ctrl-b + z        Zoom current pane (toggle fullscreen)
  Ctrl-b + d        Detach from session (dashboard keeps running)
  Ctrl-c            Stop the current pane's command

CLEANUP:

  Kill the dashboard session:
    tmux kill-session -t unity-monitor

  Re-attach to running dashboard:
    tmux attach -t unity-monitor

EOF
  exit 0
}

# Parse arguments
while (( "$#" )); do
  case "$1" in
    -h|--help)
      show_help
      ;;
    *)
      print_error "Unknown argument: $1"
      echo "Use --help for usage information."
      exit 2
      ;;
  esac
done

# Check for tmux
if ! command -v tmux &> /dev/null; then
  print_error "tmux is required but not installed."
  echo "Install with: brew install tmux"
  exit 1
fi

# Determine which process monitor to use
if command -v htop &> /dev/null; then
  PROCESS_MONITOR="htop"
else
  print_warn "htop not found, falling back to top"
  echo "Install htop for a better experience: brew install htop"
  PROCESS_MONITOR="top"
fi

# Check for nettop (macOS only)
if command -v nettop &> /dev/null; then
  NETWORK_MONITOR="nettop -P -d -J bytes_in,bytes_out"
else
  print_warn "nettop not found (macOS only)"
  echo "Falling back to netstat-based monitoring"
  NETWORK_MONITOR="watch -n 1 'echo \"=== Network Stats ===\"; netstat -an | grep -c ESTABLISHED; echo \"ESTABLISHED connections\"; netstat -an | grep -c TIME_WAIT; echo \"TIME_WAIT connections\"'"
fi

# File descriptor monitor command
FD_MONITOR='watch -n 1 '\''
echo "=== Python File Descriptors ==="
echo
total=0
for pid in $(pgrep -x python 2>/dev/null || pgrep -x python3 2>/dev/null || echo ""); do
  if [ -n "$pid" ]; then
    count=$(lsof -p "$pid" 2>/dev/null | wc -l | tr -d " ")
    cmd=$(ps -p "$pid" -o comm= 2>/dev/null || echo "python")
    echo "PID $pid ($cmd): $count FDs"
    total=$((total + count))
  fi
done
echo
if [ $total -eq 0 ]; then
  echo "(no Python processes found)"
else
  echo "────────────────────"
  echo "Total: $total FDs"
  limit=$(ulimit -n)
  echo "Limit: $limit per process"
  if [ $total -gt $((limit / 2)) ]; then
    echo "⚠️  Warning: approaching limit"
  fi
fi
'\'''

# TCP connection monitor command
TCP_MONITOR='watch -n 1 '\''
echo "=== TCP Connections ==="
echo
est=$(netstat -an 2>/dev/null | grep -c ESTABLISHED || echo 0)
tw=$(netstat -an 2>/dev/null | grep -c TIME_WAIT || echo 0)
listen=$(netstat -an 2>/dev/null | grep -c LISTEN || echo 0)
echo "ESTABLISHED: $est"
echo "TIME_WAIT:   $tw"
echo "LISTENING:   $listen"
echo
echo "────────────────────"
echo "Total active: $((est + tw))"
echo
if [ $tw -gt 100 ]; then
  echo "⚠️  High TIME_WAIT count"
  echo "   (normal after many connections close)"
fi
'\'''

# Kill existing session if it exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  print_info "Killing existing $SESSION_NAME session..."
  tmux kill-session -t "$SESSION_NAME"
fi

print_info "Launching resource monitor dashboard..."
print_info "Session name: $SESSION_NAME"
echo

# Create the tmux session with the layout
# Layout:
#   ┌─────────────────────────┐
#   │          htop           │  (45% height)
#   ├─────────────────────────┤
#   │         nettop          │  (35% height)
#   ├────────────┬────────────┤
#   │    FDs     │    TCP     │  (20% height)
#   └────────────┴────────────┘

# Create session with htop in first pane
tmux new-session -d -s "$SESSION_NAME" -n "monitor" "$PROCESS_MONITOR"

# Split horizontally for nettop (bottom half initially)
tmux split-window -v -t "$SESSION_NAME" "$NETWORK_MONITOR"

# Split the bottom pane again for the stats panes
tmux split-window -v -t "$SESSION_NAME" "bash -c '$FD_MONITOR'"

# Split the bottom-most pane horizontally
tmux split-window -h -t "$SESSION_NAME" "bash -c '$TCP_MONITOR'"

# Adjust pane sizes (top pane gets more space)
# Select pane 0 (htop) and resize
tmux select-pane -t "$SESSION_NAME:0.0"
tmux resize-pane -t "$SESSION_NAME:0.0" -y 45%

# Select pane 1 (nettop) and resize
tmux select-pane -t "$SESSION_NAME:0.1"
tmux resize-pane -t "$SESSION_NAME:0.1" -y 35%

# Select pane 0 initially (htop)
tmux select-pane -t "$SESSION_NAME:0.0"

# Print usage info
cat << EOF
Dashboard launched! Attaching now...

╔══════════════════════════════════════════════════════════════════╗
║  KEYBOARD SHORTCUTS                                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Ctrl-b + arrow    Move between panes                            ║
║  Ctrl-b + z        Zoom current pane (toggle fullscreen)         ║
║  Ctrl-b + d        Detach (dashboard keeps running in background)║
║  Ctrl-c            Stop current pane's command                   ║
╠══════════════════════════════════════════════════════════════════╣
║  TO RE-ATTACH:     tmux attach -t $SESSION_NAME                  ║
║  TO KILL:          tmux kill-session -t $SESSION_NAME            ║
╚══════════════════════════════════════════════════════════════════╝

EOF

# Attach to the session
exec tmux attach -t "$SESSION_NAME"
