#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Resource Monitor Dashboard
# ============================================================================
#
# Launches a tmux-based dashboard for monitoring system resources during
# test runs. Optimized for network I/O heavy workloads (LLM API calls, etc.)
#
# Supported Platforms:
#   - macOS (full support)
#   - Linux (full support)
#   - Windows (via WSL)
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
#   │              Network Monitor                 │
#   │         (Per-process Network I/O)            │
#   ├──────────────────────┬───────────────────────┤
#   │   File Descriptors   │   TCP Connections     │
#   │  (Python processes)  │   (Active sockets)    │
#   └──────────────────────┴───────────────────────┘
#
# Requirements:
#   - tmux (required on all platforms)
#   - htop (recommended, falls back to top)
#   - Platform-specific network tools (auto-detected)
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
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_platform() { echo -e "${CYAN}[PLATFORM]${NC} $1"; }

# ============================================================================
# OS Detection
# ============================================================================
detect_os() {
  case "$(uname -s)" in
    Darwin*)
      echo "macos"
      ;;
    Linux*)
      # Check if running in WSL
      if grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
        echo "wsl"
      else
        echo "linux"
      fi
      ;;
    CYGWIN*|MINGW*|MSYS*)
      echo "windows"
      ;;
    *)
      echo "unknown"
      ;;
  esac
}

OS_TYPE=$(detect_os)

# ============================================================================
# Help Text
# ============================================================================
show_help() {
  cat << 'EOF'
Resource Monitor Dashboard
==========================

Launches a tmux-based dashboard for monitoring system resources during
test runs. Optimized for network I/O heavy workloads.

SUPPORTED PLATFORMS:
  - macOS    Full support with native tools (nettop, lsof)
  - Linux    Full support with native tools (nethogs/iftop/ss, lsof)
  - Windows  Supported via WSL (Windows Subsystem for Linux)

USAGE:
  ./monitor_resources.sh          Launch the dashboard
  ./monitor_resources.sh --help   Show this help

DASHBOARD PANES:

  1. htop/top (top section)
     - CPU usage per core
     - Memory usage (used/cached/buffers)
     - Process list sorted by CPU
     - Swap usage

  2. Network Monitor (middle section)
     - macOS: nettop (per-process network I/O)
     - Linux: nethogs or iftop (if installed), otherwise ss stats
     - Shows bytes in/out per process

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
    - Network monitor shows per-process bandwidth
    - Look for Python processes with high bytes/sec
    - Many ESTABLISHED connections = high parallelism

  File Descriptors:
    - Each TCP connection = 1 file descriptor
    - If count approaches ulimit, you'll see failures
    - Fix: increase with 'ulimit -n 4096' before running tests

  TIME_WAIT:
    - Normal after connections close (~60s on macOS, ~30s on Linux)
    - Very high counts may indicate excessive connection churn
    - Consider connection pooling if problematic

PLATFORM-SPECIFIC TIPS:

  macOS:
    ulimit -n 4096                              # Increase FD limit
    sudo sysctl -w kern.maxfiles=65536          # Kernel tuning

  Linux:
    ulimit -n 4096                              # Increase FD limit
    sudo sysctl -w net.core.somaxconn=65535     # Kernel tuning
    sudo sysctl -w net.ipv4.tcp_tw_reuse=1      # Faster TIME_WAIT recycling

  Windows (WSL):
    Run this script from within WSL (e.g., Ubuntu)
    WSL2 recommended for better networking performance

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

INSTALLATION:

  macOS:
    brew install tmux htop

  Ubuntu/Debian:
    sudo apt install tmux htop nethogs iftop

  Fedora/RHEL:
    sudo dnf install tmux htop nethogs iftop

  Arch Linux:
    sudo pacman -S tmux htop nethogs iftop

EOF
  exit 0
}

# ============================================================================
# Parse Arguments
# ============================================================================
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

# ============================================================================
# Platform Checks
# ============================================================================
if [[ "$OS_TYPE" == "windows" ]]; then
  print_error "Native Windows is not supported."
  echo "Please run this script from WSL (Windows Subsystem for Linux)."
  echo ""
  echo "To install WSL:"
  echo "  wsl --install"
  echo ""
  echo "Then run this script from the WSL terminal."
  exit 1
fi

if [[ "$OS_TYPE" == "unknown" ]]; then
  print_warn "Unknown operating system: $(uname -s)"
  echo "Attempting to continue with Linux-like defaults..."
  OS_TYPE="linux"
fi

print_platform "Detected: $OS_TYPE"

# Check for tmux
if ! command -v tmux &> /dev/null; then
  print_error "tmux is required but not installed."
  case "$OS_TYPE" in
    macos)
      echo "Install with: brew install tmux"
      ;;
    linux|wsl)
      echo "Install with: sudo apt install tmux  (Debian/Ubuntu)"
      echo "          or: sudo dnf install tmux  (Fedora/RHEL)"
      echo "          or: sudo pacman -S tmux    (Arch)"
      ;;
  esac
  exit 1
fi

# ============================================================================
# Process Monitor (htop/top)
# ============================================================================
if command -v htop &> /dev/null; then
  PROCESS_MONITOR="htop"
else
  print_warn "htop not found, falling back to top"
  case "$OS_TYPE" in
    macos)
      echo "Install htop for a better experience: brew install htop"
      ;;
    linux|wsl)
      echo "Install htop for a better experience: sudo apt install htop"
      ;;
  esac
  PROCESS_MONITOR="top"
fi

# ============================================================================
# Network Monitor (platform-specific)
# ============================================================================
setup_network_monitor() {
  case "$OS_TYPE" in
    macos)
      if command -v nettop &> /dev/null; then
        NETWORK_MONITOR="nettop -P -d -J bytes_in,bytes_out"
      else
        # Fallback for older macOS or missing nettop
        NETWORK_MONITOR="watch -n 1 'echo \"=== Network Stats (netstat) ===\"; echo; netstat -an | grep -c ESTABLISHED | xargs -I{} echo \"ESTABLISHED: {}\"; netstat -an | grep -c TIME_WAIT | xargs -I{} echo \"TIME_WAIT: {}\"'"
      fi
      ;;
    linux|wsl)
      if command -v nethogs &> /dev/null; then
        # nethogs needs root, check if we can use it
        if [[ $EUID -eq 0 ]] || command -v sudo &> /dev/null; then
          NETWORK_MONITOR="sudo nethogs -d 1"
          print_info "Using nethogs for per-process network monitoring (may require sudo password)"
        else
          NETWORK_MONITOR=""
        fi
      fi

      if [[ -z "${NETWORK_MONITOR:-}" ]] && command -v iftop &> /dev/null; then
        if [[ $EUID -eq 0 ]] || command -v sudo &> /dev/null; then
          NETWORK_MONITOR="sudo iftop -t -s 1"
          print_info "Using iftop for network monitoring (may require sudo password)"
        fi
      fi

      if [[ -z "${NETWORK_MONITOR:-}" ]]; then
        # Fallback to ss-based monitoring (no sudo needed)
        print_warn "nethogs/iftop not found, using socket statistics"
        echo "Install for better network monitoring: sudo apt install nethogs iftop"
        NETWORK_MONITOR='watch -n 1 '\''
echo "=== Network Socket Stats (ss) ==="
echo
echo "TCP Sockets by State:"
ss -tan 2>/dev/null | tail -n +2 | awk "{print \$1}" | sort | uniq -c | sort -rn
echo
echo "─────────────────────────────────"
echo "Top Processes with Network Sockets:"
ss -tanp 2>/dev/null | grep -oP "users:\(\(\"\K[^\"]+|pid=\K[0-9]+" | paste - - | sort | uniq -c | sort -rn | head -10
'\'''
      fi
      ;;
  esac
}

setup_network_monitor


# ============================================================================
# Launch Dashboard
# ============================================================================

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
#   │          htop           │  (40% height)
#   ├─────────────────────────┤
#   │     Network Monitor     │  (35% height)
#   ├────────────┬────────────┤
#   │    FDs     │    TCP     │  (25% height)
#   └────────────┴────────────┘

# FD monitor command - inline script avoids quoting issues
FD_CMD='while true; do clear; echo "=== Python File Descriptors ==="; echo; total=0; for pid in $(pgrep python 2>/dev/null); do if [ -n "$pid" ]; then if [ -d "/proc/$pid/fd" ]; then count=$(ls /proc/$pid/fd 2>/dev/null | wc -l); else count=$(lsof -p "$pid" 2>/dev/null | wc -l); fi; cmd=$(ps -p "$pid" -o comm= 2>/dev/null || echo python); echo "PID $pid ($cmd): $count FDs"; total=$((total + count)); fi; done; echo; if [ $total -eq 0 ]; then echo "(no Python processes)"; else echo "────────────────────"; echo "Total: $total FDs"; echo "Limit: $(ulimit -n) per process"; fi; sleep 1; done'

# TCP monitor command - inline script
TCP_CMD='while true; do clear; echo "=== TCP Connections ==="; echo; if command -v ss >/dev/null 2>&1; then est=$(ss -tan state established 2>/dev/null | tail -n +2 | wc -l); tw=$(ss -tan state time-wait 2>/dev/null | tail -n +2 | wc -l); listen=$(ss -tln 2>/dev/null | tail -n +2 | wc -l); else est=$(netstat -an 2>/dev/null | grep -c ESTABLISHED || echo 0); tw=$(netstat -an 2>/dev/null | grep -c TIME_WAIT || echo 0); listen=$(netstat -an 2>/dev/null | grep -c LISTEN || echo 0); fi; echo "ESTABLISHED: $est"; echo "TIME_WAIT:   $tw"; echo "LISTENING:   $listen"; echo; echo "────────────────────"; echo "Total active: $((est + tw))"; sleep 1; done'

# Create session with htop in first pane (pane 0)
tmux new-session -d -s "$SESSION_NAME" -n "monitor" "$PROCESS_MONITOR"

# Split pane 0 vertically -> creates pane 1 (network) below pane 0 (htop)
tmux split-window -v -t "$SESSION_NAME:0" "$NETWORK_MONITOR"

# Split pane 1 vertically -> creates pane 2 (FD) below pane 1 (network)
tmux split-window -v -t "$SESSION_NAME:0.1" "bash -c '$FD_CMD'"

# Split pane 2 horizontally -> creates pane 3 (TCP) to the right of pane 2 (FD)
tmux split-window -h -t "$SESSION_NAME:0.2" "bash -c '$TCP_CMD'"

# Set layout: main-horizontal with htop on top, rest below
# Then manually adjust sizes for better proportions
tmux select-pane -t "$SESSION_NAME:0.0"
tmux resize-pane -t "$SESSION_NAME:0.0" -y 18

tmux select-pane -t "$SESSION_NAME:0.1"
tmux resize-pane -t "$SESSION_NAME:0.1" -y 15

# Select htop pane initially
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
