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
#   │                   (70%)                      │
#   ├──────────────┬──────────────┬────────────────┤
#   │   Network    │     FDs      │      TCP       │
#   │    (33%)     │    (33%)     │     (33%)      │
#   └──────────────┴──────────────┴────────────────┘
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

# ---- Increase file descriptor limit ----
# Match the limit set by parallel_run.sh so the FD monitor displays accurate
# limits. Without this, running the monitor from a different terminal would
# show macOS's default 256, which doesn't reflect the actual limits of the
# pytest processes (which inherit 8192 from parallel_run.sh).
ulimit -n 8192 2>/dev/null || true

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
      # Custom network summary - fits in narrow terminals
      NETWORK_MONITOR='bash -c "set +e; prev_in=0; prev_out=0; while true; do clear; echo \"=== Network Activity ===\"; echo; stats=\$(netstat -ib 2>/dev/null | grep -v \"^Name\" | grep -v \"^lo\" | head -5); curr_in=\$(echo \"\$stats\" | awk \"{sum+=\\\$7} END {print sum+0}\"); curr_out=\$(echo \"\$stats\" | awk \"{sum+=\\\$10} END {print sum+0}\"); if [ \$prev_in -gt 0 ]; then delta_in=\$(( (curr_in - prev_in) / 2 )); delta_out=\$(( (curr_out - prev_out) / 2 )); echo \"Throughput (2s avg):\"; if [ \$delta_in -gt 1048576 ]; then echo \"  IN:  \$(( delta_in / 1048576 )) MB/s\"; elif [ \$delta_in -gt 1024 ]; then echo \"  IN:  \$(( delta_in / 1024 )) KB/s\"; else echo \"  IN:  \$delta_in B/s\"; fi; if [ \$delta_out -gt 1048576 ]; then echo \"  OUT: \$(( delta_out / 1048576 )) MB/s\"; elif [ \$delta_out -gt 1024 ]; then echo \"  OUT: \$(( delta_out / 1024 )) KB/s\"; else echo \"  OUT: \$delta_out B/s\"; fi; else echo \"Throughput: measuring...\"; fi; prev_in=\$curr_in; prev_out=\$curr_out; echo; echo \"────────────────────────\"; echo; py_conns=\$(lsof -i -n 2>/dev/null | grep -c python || echo 0); echo \"Python connections: \$py_conns\"; echo; echo \"Top processes:\"; lsof -i -n 2>/dev/null | awk \"{print \\\$1}\" | sort | uniq -c | sort -rn | head -5 | awk \"{printf \\\"  %-12s %s\\n\\\", \\\$2, \\\$1}\"; sleep 2; done"'
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
#   ┌─────────────────────────────────────────┐
#   │                  htop                   │  (70% height)
#   ├─────────────┬─────────────┬─────────────┤
#   │   Network   │     FDs     │     TCP     │  (30% height, 33% each)
#   └─────────────┴─────────────┴─────────────┘

# FD monitor command - shows summary only (not per-process) to reduce visual noise
# Note: Collects all data BEFORE clearing screen to avoid flicker from slow lsof
# Warning triggers when average FDs per process exceeds 50% of the per-process limit
FD_CMD='set +e; while true; do total=0; proc_count=0; max_fd=0; pids=$(pgrep -i python 2>/dev/null || true); for pid in $pids; do if [ -n "$pid" ]; then proc_count=$((proc_count + 1)); if [ -d "/proc/$pid/fd" ]; then count=$(ls /proc/$pid/fd 2>/dev/null | wc -l | tr -d " "); else count=$(lsof -p "$pid" 2>/dev/null | wc -l | tr -d " "); fi; count=${count:-0}; total=$((total + count)); if [ $count -gt $max_fd ]; then max_fd=$count; fi; fi; done; limit=$(ulimit -n); clear; echo "=== File Descriptors ==="; echo; if [ "$proc_count" -eq 0 ]; then echo "(no Python procs)"; else echo "Procs:        $proc_count"; echo "FDs (total):  $total"; echo; echo "Max (found):  $max_fd/proc"; echo "Limit:        $limit/proc"; if [ $max_fd -gt $((limit * 3 / 4)) ]; then echo; echo "⚠️  Near limit!"; fi; fi; sleep 2; done'

# TCP monitor command - inline script
# Note: set +e disables exit-on-error; grep -c returns 1 on no match which would crash without this
TCP_CMD='set +e; while true; do clear; echo "=== TCP Connections ==="; echo; if command -v ss >/dev/null 2>&1; then est=$(ss -tan state established 2>/dev/null | tail -n +2 | wc -l | tr -d " "); tw=$(ss -tan state time-wait 2>/dev/null | tail -n +2 | wc -l | tr -d " "); listen=$(ss -tln 2>/dev/null | tail -n +2 | wc -l | tr -d " "); else est=$(netstat -an 2>/dev/null | grep -c ESTABLISHED); est=${est:-0}; tw=$(netstat -an 2>/dev/null | grep -c TIME_WAIT); tw=${tw:-0}; listen=$(netstat -an 2>/dev/null | grep -c LISTEN); listen=${listen:-0}; fi; est=${est:-0}; tw=${tw:-0}; listen=${listen:-0}; echo "ESTAB: $est"; echo "T_WAIT: $tw"; echo "LISTEN: $listen"; echo "────────────"; echo "Active: $((est + tw))"; sleep 1; done'

# Create session with htop in first pane (pane 0)
tmux new-session -d -s "$SESSION_NAME" -n "monitor" "$PROCESS_MONITOR"

# Split pane 0 vertically -> creates pane 1 (bottom row) below pane 0 (htop)
tmux split-window -v -t "$SESSION_NAME:0" "$NETWORK_MONITOR"

# Split pane 1 horizontally -> creates pane 2 (FDs) to the right of Network
tmux split-window -h -t "$SESSION_NAME:0.1" "bash -c '$FD_CMD'"

# Split pane 2 horizontally -> creates pane 3 (TCP) to the right of FDs
tmux split-window -h -t "$SESSION_NAME:0.2" "bash -c '$TCP_CMD'"

# Resize: htop gets 70% vertical, bottom row gets 30%
tmux resize-pane -t "$SESSION_NAME:0.0" -y 70%

# Make the three bottom panes roughly equal width (33% each)
# Pane 1 (Network) gets 33% of window width
tmux resize-pane -t "$SESSION_NAME:0.1" -x 33%
# Pane 2 (FDs) gets 50% of remaining (which is ~33% of total)
tmux resize-pane -t "$SESSION_NAME:0.2" -x 50%
# Pane 3 (TCP) automatically gets the rest (~33%)

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
