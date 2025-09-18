#!/usr/bin/env bash
set -euo pipefail

# 1) Enable Screen Sharing (VNC) and set a VNC password
# Note: Requires sudo privileges. Apple VNC passwords are limited to 8 characters.
# If UNIFY_KEY provided, use first 8 chars. Otherwise use VNC_PASSWORD or default.
if [[ -n "${UNIFY_KEY:-}" ]]; then
  VNC_PASSWORD=${UNIFY_KEY:0:8}
else
  VNC_PASSWORD=${VNC_PASSWORD:-"changeme"}
  VNC_PASSWORD=${VNC_PASSWORD:0:8}
fi

echo "Enabling Screen Sharing..."
sudo /System/Library/CoreServices/RemoteManagement/ARDAgent.app/Contents/Resources/kickstart \
  -activate -configure -access -on \
  -clientopts -setvnclegacy -vnclegacy yes \
  -clientopts -setvncpw -vncpw "$VNC_PASSWORD" \
  -restart -agent -privs -all

# 2) Screen Sharing (Apple VNC) listens on 5900 by default
VNC_HOST=localhost
VNC_PORT=${VNC_PORT:-5900}

# 3) Ensure noVNC assets exist
NOVNC_DIR=/opt/novnc
if [[ ! -d "$NOVNC_DIR" ]]; then
  echo "Error: $NOVNC_DIR not found. Please run desktop/mac/install.sh first." >&2
  exit 1
fi

# 4) Determine websockify executable
if command -v websockify >/dev/null 2>&1; then
  WEBSOCKIFY_BIN=$(command -v websockify)
elif command -v python3 >/dev/null 2>&1; then
  WEBSOCKIFY_BIN="python3 -m websockify"
elif command -v python >/dev/null 2>&1; then
  WEBSOCKIFY_BIN="python -m websockify"
else
  echo "Error: websockify not found. Install via Python pip (see desktop/mac/install.sh)." >&2
  exit 1
fi

# 5) Start noVNC web proxy on 6080 -> 5900
NOVNC_PORT=${NOVNC_PORT:-6080}
PID_FILE=/tmp/novnc_websockify.pid

# 5a) Clean shutdown handler
cleanup() {
  echo "\nShutting down noVNC/websockify..."
  if [[ -n "${TS_PID:-}" ]] && kill -0 "${TS_PID}" 2>/dev/null; then
    kill "${TS_PID}" 2>/dev/null || true
    sleep 0.5
    if kill -0 "${TS_PID}" 2>/dev/null; then
      kill -9 "${TS_PID}" 2>/dev/null || true
    fi
  fi
  if [[ -f "${PID_FILE}" ]]; then
    WS_PID_FILE=$(cat "${PID_FILE}" 2>/dev/null || echo "")
    if [[ -n "${WS_PID_FILE}" ]] && kill -0 "${WS_PID_FILE}" 2>/dev/null; then
      kill "${WS_PID_FILE}" 2>/dev/null || true
      sleep 0.5
      if kill -0 "${WS_PID_FILE}" 2>/dev/null; then
        kill -9 "${WS_PID_FILE}" 2>/dev/null || true
      fi
    fi
    rm -f "${PID_FILE}"
  fi

  # Disable Screen Sharing on teardown (inline from cleanup.sh)
  echo "Disabling Screen Sharing..."
  sudo -n /System/Library/CoreServices/RemoteManagement/ARDAgent.app/Contents/Resources/kickstart -deactivate -stop || true
  sudo -n /System/Library/CoreServices/RemoteManagement/ARDAgent.app/Contents/Resources/kickstart -configure -access -off || true
  sudo -n /System/Library/CoreServices/RemoteManagement/ARDAgent.app/Contents/Resources/kickstart -configure -clientopts -setvnclegacy -vnclegacy no || true

  echo "Unloading Screen Sharing daemon..."
  sudo -n launchctl unload -w /System/Library/LaunchDaemons/com.apple.screensharing.plist || true

  echo "Cleanup complete."
}

trap cleanup INT TERM EXIT

# If an existing websockify is running on the same port, stop it
if lsof -iTCP -sTCP:LISTEN -P | grep -E ":${NOVNC_PORT}.*(websockify|Python)" >/dev/null 2>&1; then
  echo "Stopping existing websockify on port ${NOVNC_PORT}..."
  pkill -f "websockify.*${NOVNC_PORT}" || true
  pkill -f "python(3)? -m websockify .*${NOVNC_PORT}" || true
  sleep 0.5
fi

echo "Starting noVNC web proxy on http://localhost:${NOVNC_PORT}/vnc.html (to ${VNC_HOST}:${VNC_PORT})"
set +e
nohup bash -c "${WEBSOCKIFY_BIN} --web='${NOVNC_DIR}' ${NOVNC_PORT} ${VNC_HOST}:${VNC_PORT}" >/tmp/novnc_websockify.log 2>&1 &
WS_PID=$!
set -e
echo ${WS_PID} > ${PID_FILE}
echo "websockify started with PID ${WS_PID}. Logs: /tmp/novnc_websockify.log"

echo "Done. Open: http://localhost:${NOVNC_PORT}/vnc.html"

# Start magnitude agent-service (ts-node) like linux/remote.sh
echo "Starting magnitude agent-service..."
npx ts-node agent-service/src/index.ts &
TS_PID=$!

# Keep the script in the foreground to handle clean shutdown on Ctrl+C
echo "Press Ctrl+C to stop."
wait "${TS_PID}" || true
