#!/usr/bin/env bash
set -euo pipefail

# Use the existing desktop display (default to :0 if not set)
DISPLAY="${DISPLAY:-:0}"

# Export UNIFY_KEY from first argument if provided, otherwise require env var
if [[ ${1:-} != "" ]]; then
UNIFY_KEY="$1"
fi
export UNIFY_KEY
if [[ -z "${UNIFY_KEY:-}" ]]; then
echo "Error: UNIFY_KEY not provided. Pass as first argument or set env UNIFY_KEY." >&2
exit 1
fi

# If x11vnc is already running, terminate it first
if pgrep -x x11vnc >/dev/null 2>&1; then
echo "Found existing x11vnc process. Terminating..."
pkill -x x11vnc || true
# Wait briefly for shutdown
for i in {1..10}; do
if pgrep -x x11vnc >/dev/null 2>&1; then
sleep 0.3
else
break
fi
done
fi

# Start x11vnc (background)
x11vnc -display "$DISPLAY" -nopw -forever -shared -bg -rfbport 5900 -passwd "$UNIFY_KEY" \
       -rfbportv6 0 -noxdamage -nowf -nocursorshape -cursor arrow -nodpms
# Capture the newest x11vnc PID
X11VNC_PID="$(pgrep -n x11vnc || true)"

# Start the noVNC web proxy to expose VNC on http://localhost:6080/vnc.html
websockify --web=/opt/novnc 6080 localhost:5900 &
WEBSOCKIFY_PID=$!

# Graceful shutdown
cleanup() {
  echo "[remote] Shutting down..."
  if [[ -n "${TS_PID:-}" ]]; then
    kill -TERM "$TS_PID" 2>/dev/null || true
    wait "$TS_PID" 2>/dev/null || true
  fi
  if [[ -n "${WEBSOCKIFY_PID:-}" ]]; then
    kill -TERM "$WEBSOCKIFY_PID" 2>/dev/null || true
    wait "$WEBSOCKIFY_PID" 2>/dev/null || true
  fi
  if [[ -n "${X11VNC_PID:-}" ]]; then
    kill -TERM "$X11VNC_PID" 2>/dev/null || true
    wait "$X11VNC_PID" 2>/dev/null || true
  fi
}

trap cleanup SIGTERM SIGINT

npx ts-node agent-service/src/index.ts &
TS_PID=$!

wait "$TS_PID"
cleanup
