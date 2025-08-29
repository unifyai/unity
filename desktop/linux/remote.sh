#!/usr/bin/env bash
set -euo pipefail

# Use the existing desktop display (default to :99 if not set)
DISPLAY="${DISPLAY:-:99}"

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

x11vnc -display "$DISPLAY" -nopw -forever -shared -bg -rfbport 5900 -passwd "$UNIFY_KEY" \
       -rfbportv6 0 -noxdamage -nowf -nocursorshape -cursor arrow -nodpms

# Start the noVNC web proxy to expose VNC on http://localhost:6080/vnc.html
websockify --web=/opt/novnc 6080 localhost:5900 &

npx ts-node agent-service/src/index.ts
