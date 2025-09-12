#!/usr/bin/env bash
set -euo pipefail

# Start a free, ephemeral Cloudflare Tunnel ("trycloudflare") to localhost:3000 only.
# This script is intentionally fixed to tunnel http://localhost:3000.

LOCAL_PORT=3000

echo "[tunnel] Target: http://localhost:${LOCAL_PORT}"

# Ensure cloudflared exists (install via Homebrew if missing)
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "[tunnel] cloudflared not found. Installing via Homebrew..."
  if ! command -v brew >/dev/null 2>&1; then
    echo "[tunnel] Error: Homebrew not found. Install Homebrew or cloudflared manually: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" >&2
    exit 1
  fi
  brew update
  brew install cloudflared
  echo "[tunnel] cloudflared installed. Version: $(cloudflared --version | head -n1)"
else
  echo "[tunnel] cloudflared present: $(cloudflared --version | head -n1)"
fi

# Clean shutdown
CF_PID=""
cleanup() {
  echo "\n[tunnel] Shutting down tunnel..."
  if [[ -n "$CF_PID" ]] && kill -0 "$CF_PID" 2>/dev/null; then
    kill "$CF_PID" 2>/dev/null || true
    sleep 0.5
    if kill -0 "$CF_PID" 2>/dev/null; then
      kill -9 "$CF_PID" 2>/dev/null || true
    fi
  fi
}
trap cleanup INT TERM EXIT

echo "[tunnel] Starting trycloudflare (no login required)." \
     "Press Ctrl+C to stop."

# cloudflared prints the public URL in logs; capture and surface it.
LOG_FILE="/tmp/trycloudflare_${LOCAL_PORT}.log"
: > "$LOG_FILE"

set +e
cloudflared tunnel --url "http://localhost:${LOCAL_PORT}" 2>&1 | tee "$LOG_FILE" &
CF_PID=$!
set -e

# Try to extract the URL quickly
tries=20
url=""
while (( tries-- > 0 )); do
  if grep -Eo 'https://[a-zA-Z0-9.-]+trycloudflare\.com' "$LOG_FILE" >/dev/null 2>&1; then
    url=$(grep -Eo 'https://[a-zA-Z0-9.-]+trycloudflare\.com' "$LOG_FILE" | head -n1)
    break
  fi
  # Cloudflared newer logs may print a generic https URL too
  if grep -Eo 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$LOG_FILE" >/dev/null 2>&1; then
    url=$(grep -Eo 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$LOG_FILE" | head -n1)
    break
  fi
  sleep 0.3
done

if [[ -n "$url" ]]; then
  echo "[tunnel] Public URL: $url"
else
  echo "[tunnel] Waiting for public URL... check logs: $LOG_FILE"
fi

wait "$CF_PID" || true
