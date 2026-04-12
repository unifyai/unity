#!/bin/bash

# Exit on any error
set -e

# Record container start time (milliseconds since epoch).
# Used by the Python process to compute container spin-up duration (U1).
export CONTAINER_START_TIME_MS=$(date +%s%3N)

# Global variables to track processes
MAIN_PID=""
AGENT_PID=""
WATCHDOG_PID=""
DISPLAY_PID=""
DEVICE_PID=""

uptime_ms() {
    now_ms=$(date +%s%3N)
    echo $((now_ms - CONTAINER_START_TIME_MS))
}

shutdown_reason() {
    if [ -f /tmp/oom_prevention_shutdown ]; then
        echo "oom_prevention"
    else
        echo "external_sigterm"
    fi
}

# Monitor cgroup memory usage and send SIGTERM before the kernel OOM-kills us.
# Runs as a background process with negligible overhead (reads a sysfs file
# every few seconds).  When usage crosses the threshold, it writes a marker
# file so the Python shutdown path can log the reason, then sends SIGTERM to
# the main process — triggering the normal graceful shutdown chain.
memory_watchdog() {
    if [ -f /sys/fs/cgroup/memory.max ]; then
        max_file=/sys/fs/cgroup/memory.max
        current_file=/sys/fs/cgroup/memory.current
    elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        max_file=/sys/fs/cgroup/memory/memory.limit_in_bytes
        current_file=/sys/fs/cgroup/memory/memory.usage_in_bytes
    else
        echo "[MEMORY_WATCHDOG] No cgroup memory files found, watchdog disabled"
        return
    fi

    max_bytes=$(cat "$max_file")
    if [ "$max_bytes" = "max" ]; then
        echo "[MEMORY_WATCHDOG] No memory limit set, watchdog disabled"
        return
    fi

    threshold_pct=${MEMORY_WATCHDOG_THRESHOLD:-90}
    interval=${MEMORY_WATCHDOG_INTERVAL:-5}
    threshold_bytes=$((max_bytes * threshold_pct / 100))

    echo "[MEMORY_WATCHDOG] Limit: $((max_bytes / 1048576))MiB, threshold: ${threshold_pct}% ($((threshold_bytes / 1048576))MiB), check every ${interval}s"

    while kill -0 "$MAIN_PID" 2>/dev/null; do
        sleep "$interval"
        current_bytes=$(cat "$current_file" 2>/dev/null) || continue
        if [ "$current_bytes" -ge "$threshold_bytes" ]; then
            pct=$((current_bytes * 100 / max_bytes))
            echo "$(date '+%Y-%m-%d %H:%M:%S.%3N') - [MEMORY_WATCHDOG] Usage at ${pct}% ($((current_bytes / 1048576))/$((max_bytes / 1048576))MiB) — triggering graceful shutdown"
            touch /tmp/oom_prevention_shutdown
            kill -TERM "$MAIN_PID" 2>/dev/null || true
            return
        fi
    done
}

stop_agent_service() {
    # Prefer PID file written by Python after an in-flight restart
    # (the original $AGENT_PID becomes stale when Python restarts the service).
    local pid_file="/tmp/agent-service.pid"
    local pid=""
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
    elif [ ! -z "$AGENT_PID" ]; then
        pid=$AGENT_PID
    fi

    if [ ! -z "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping agent-service (PID: $pid)..."
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    else
        echo "Stopping agent-service..."
        pkill -f "ts-node" 2>/dev/null || true
    fi
}

# Signal handler: forward SIGTERM to the Python process and wait for it to
# finish its own shutdown sequence (which now includes the GCS log upload).
on_signal() {
    local reason
    reason=$(shutdown_reason)
    echo "$(date '+%Y-%m-%d %H:%M:%S.%3N') - [ENTRYPOINT] Received shutdown signal (reason=${reason}, uptime_ms=$(uptime_ms), main_pid=${MAIN_PID:-none}, agent_pid=${AGENT_PID:-none}), cleaning up..."

    if [ ! -z "$WATCHDOG_PID" ]; then
        kill $WATCHDOG_PID 2>/dev/null || true
    fi

    if [ ! -z "$MAIN_PID" ]; then
        echo "Stopping main application (PID: $MAIN_PID)..."
        kill -TERM $MAIN_PID 2>/dev/null || true
        wait $MAIN_PID 2>/dev/null || true
    fi

    stop_agent_service

    if [ ! -z "$DEVICE_PID" ]; then
        kill $DEVICE_PID 2>/dev/null || true
    fi
    if [ ! -z "$DISPLAY_PID" ]; then
        kill $DISPLAY_PID 2>/dev/null || true
    fi

    echo "Cleanup complete"
    exit 0
}

trap on_signal SIGTERM SIGINT

# Create log directories for file-based traces in background
mkdir -p /var/log/unity /var/log/unify /var/log/unillm &

# Announce where logs will be preserved after shutdown
if [ ! -z "$UNITY_CONVERSATION_JOB_NAME" ]; then
    _GCS_BUCKET="${GCS_LOG_BUCKET:-unity-pod-logs}"
    # Derive namespace from job name suffix
    case "$UNITY_CONVERSATION_JOB_NAME" in
        *-preview)    _NS="preview" ;;
        *-staging)    _NS="staging" ;;
        *-production) _NS="production" ;;
        *)            _NS="unknown" ;;
    esac
    _GCS_PATH="gs://${_GCS_BUCKET}/${_NS}/${UNITY_CONVERSATION_JOB_NAME}/"
    echo "═══════════════════════════════════════════════════════════"
    echo "  Pod logs will be uploaded on shutdown to:"
    echo "  ${_GCS_PATH}"
    echo "  (auto-deleted after 7 days)"
    echo "═══════════════════════════════════════════════════════════"
fi

# Seed the emptyDir-backed /tmp with pre-downloaded HuggingFace models in background.
# The Dockerfile bakes models into /opt/hf-cache (user-agnostic); at runtime
# HF_HOME=/tmp/huggingface redirects lookups to the writable emptyDir volume.
if [ -d /opt/hf-cache ] && [ ! -d /tmp/huggingface ]; then
    cp -r /opt/hf-cache /tmp/huggingface &
fi

# ── Desktop stack (virtual display + audio devices) ──────────────────────────
# Reuses deploy/desktop/display.sh and deploy/desktop/device.sh to provide the
# same TigerVNC + XFCE + PipeWire/PulseAudio environment as the desktop container.
# Required for non-headless browser sessions (e.g. Google Meet join).
export XDG_RUNTIME_DIR=/tmp/runtime-unity
mkdir -p "$XDG_RUNTIME_DIR" 2>/dev/null || true
chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true

dbus-daemon --system --fork 2>/dev/null || true
eval "$(dbus-launch)"
export DBUS_SESSION_BUS_ADDRESS

echo "⬥ Starting virtual display..."
bash /app/deploy/desktop/display.sh &
DISPLAY_PID=$!

echo "⬥ Starting virtual audio devices..."
bash /app/deploy/desktop/device.sh &
DEVICE_PID=$!

sleep 3

# agent-service is started by the Python process after StartupEvent
# (see event_handlers.py _restart_agent_service_with_key)

# Start the main application
echo "⬥ Starting convo manager..."
python3 unity/conversation_manager/main.py &
MAIN_PID=$!
echo "⬥ Main application started with PID: $MAIN_PID"

# Start memory watchdog in background
memory_watchdog &
WATCHDOG_PID=$!

# Wait for the main process to exit (inactivity timeout or other self-initiated shutdown).
# The SIGTERM path is handled by the on_signal trap above and never reaches here.
MAIN_EXIT_CODE=0
wait $MAIN_PID || MAIN_EXIT_CODE=$?
echo "$(date '+%Y-%m-%d %H:%M:%S.%3N') - [ENTRYPOINT] Main process exited (code=${MAIN_EXIT_CODE}, uptime_ms=$(uptime_ms))"
kill $WATCHDOG_PID 2>/dev/null || true
stop_agent_service
kill $DEVICE_PID 2>/dev/null || true
kill $DISPLAY_PID 2>/dev/null || true
