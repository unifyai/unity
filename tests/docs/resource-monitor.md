# Resource Monitor Dashboard (`monitor_resources.sh`)

When running parallel tests with heavy network I/O (like LLM API calls), it's essential to monitor system resources. This dashboard helps you:

- **Detect bottlenecks**: Is CPU, memory, or network the limiting factor?
- **Spot connection leaks**: Are file descriptors growing unbounded?
- **Avoid hitting OS limits**: Are you approaching `ulimit` thresholds?
- **Understand test behavior**: How much network traffic are tests generating?

## Quick Start

```bash
# Launch the dashboard
monitor_resources

# Or with full path
tests/monitor_resources.sh
```

---

## Supported Platforms

| Platform | Support | Notes |
|----------|---------|-------|
| **macOS** | ✅ Full | Uses native tools (`lsof`, `netstat`) |
| **Linux** | ✅ Full | Uses `/proc` filesystem and `ss` for efficiency |
| **Windows** | ✅ Via WSL | Run from WSL terminal (Ubuntu recommended) |

---

## Dashboard Layout

The dashboard displays four panes in a tmux session:

```
┌──────────────────────────────────────────────┐
│                    htop                      │
│          (CPU, Memory, Processes)            │
│                   (70%)                      │
├──────────────┬──────────────┬────────────────┤
│   Network    │     FDs      │      TCP       │
│    (33%)     │    (33%)     │     (33%)      │
└──────────────┴──────────────┴────────────────┘
```

---

## What Each Pane Shows

### 1. htop (Top, 70% height) — System Overview

**What it displays:**
- CPU usage per core (bar graphs)
- Memory and swap usage
- Process list sorted by resource consumption
- Load average and uptime

**What to look for during tests:**

| Metric | Healthy | Warning Signs |
|--------|---------|---------------|
| CPU per core | 20-50% | All cores at 100% = CPU bottleneck |
| Memory | <80% used | Continuous growth = memory leak |
| Load average | Below core count | Exceeds core count = overloaded |
| Top processes | Python, redis | Unexpected processes consuming resources |

**Why it matters:** Parallel tests spawn many Python processes. If CPU is maxed out, tests will slow down. If memory is exhausted, the OS will start killing processes.

### 2. Network Activity (Bottom Left) — Throughput & Connections

**What it displays:**
- Network throughput (bytes/sec in and out)
- Number of Python network connections
- Top processes by connection count

**Example output:**
```
=== Network Activity ===

Throughput (2s avg):
  IN:  45 KB/s
  OUT: 12 KB/s

────────────────────────

Python connections: 190

Top processes:
  python3      190
  redis-ser    45
```

**What to look for during tests:**

| Metric | Healthy | Warning Signs |
|--------|---------|---------------|
| Throughput | Varies with test load | Drops to 0 during active tests = network issue |
| Python connections | Proportional to parallel tests | Growing unbounded = connection leak |

**Why it matters:** LLM API tests are network-bound. Low throughput with high CPU suggests inefficient connection handling. Many Python connections indicate high parallelism.

### 3. File Descriptors (Bottom Middle) — Resource Limits

**What it displays:**
- Number of Python processes
- Total file descriptors (FDs) open across all Python processes
- Per-process FD limit (`ulimit -n`)
- Warning if approaching limit

**Example output:**
```
=== File Descriptors ===

Procs: 81
FDs:   16413
Limit: 256

⚠️ Near limit!
```

**What to look for during tests:**

| Metric | Healthy | Warning Signs |
|--------|---------|---------------|
| Total FDs | Well below (procs × limit) | "Near limit!" warning |
| FDs per process | <200 (if limit is 256) | Processes failing with "Too many open files" |

**Why it matters:** Each network connection, open file, and pipe consumes one FD. macOS defaults to 256 FDs per process. With 81 Python processes averaging 200 FDs each, some are near the limit!

**Fix:** `parallel_run` automatically sets `ulimit -n 4096`, so this is handled for you. If running pytest directly, run `ulimit -n 4096` first.

### 4. TCP Connections (Bottom Right) — Socket States

**What it displays:**
- ESTABLISHED: Active, open connections
- TIME_WAIT: Recently closed, waiting for cleanup
- LISTENING: Server sockets awaiting connections
- Total active connections

**Example output:**
```
=== TCP Connections ===

ESTAB:  190
T_WAIT: 2
LISTEN: 189
────────────
Active: 192
```

**What to look for during tests:**

| Metric | Healthy | Warning Signs |
|--------|---------|---------------|
| ESTABLISHED | Proportional to active tests | Drops to 0 unexpectedly = connection failures |
| TIME_WAIT | <100 | Hundreds/thousands = excessive connection churn |
| LISTENING | Stable count | Unexpected growth = resource leak |

**Why it matters:**
- **ESTABLISHED** connections show how many API calls are in flight
- **TIME_WAIT** connections linger for 30-60 seconds after closing; high counts suggest you're opening/closing too many connections (consider connection pooling)
- **LISTENING** sockets are servers (redis, test fixtures); should stay constant

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl-b + arrow` | Move between panes |
| `Ctrl-b + z` | Zoom current pane (toggle fullscreen) |
| `Ctrl-b + d` | Detach from session (keeps running in background) |
| `Ctrl-c` | Stop the current pane's command |

---

## Managing the Dashboard

```bash
# Re-attach to a running dashboard
tmux attach -t unity-monitor

# Kill the dashboard
tmux kill-session -t unity-monitor

# Check if dashboard is running
tmux has-session -t unity-monitor && echo "Running"
```

---

## Pre-Test Tuning

**File descriptor limit:** `parallel_run` automatically sets `ulimit -n 4096` before spawning test sessions. If you're running pytest directly, increase the limit manually:

```bash
# Increase FD limit (resets on terminal close)
ulimit -n 4096

# Verify the change
ulimit -n
```

For extreme parallelism (hundreds of concurrent connections):

```bash
# macOS kernel tuning (requires sudo, resets on reboot)
sudo sysctl -w kern.maxfiles=65536
sudo sysctl -w kern.maxfilesperproc=65536

# Linux kernel tuning (requires sudo, resets on reboot)
sudo sysctl -w net.core.somaxconn=65535
sudo sysctl -w net.ipv4.tcp_tw_reuse=1
```

---

## Installation

**macOS:**

```bash
brew install tmux htop
```

**Ubuntu/Debian:**

```bash
sudo apt install tmux htop
```

**Fedora/RHEL:**

```bash
sudo dnf install tmux htop
```

**Windows (WSL):**

```powershell
# Install WSL if needed
wsl --install

# Then from within WSL (e.g., Ubuntu)
sudo apt install tmux htop
```

---

## Requirements

| Tool | Required | Notes |
|------|----------|-------|
| `tmux` | ✅ Yes | Session manager for the dashboard |
| `htop` | Recommended | Falls back to `top` if missing |
