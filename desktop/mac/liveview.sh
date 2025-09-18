#!/usr/bin/env bash
set -euo pipefail

# Cloudflare Tunnel helper for macOS.
# - If no hostname is provided → start a free, ephemeral tunnel to http://localhost:6080 (trycloudflare).
# - If a hostname is provided (arg or TUNNEL_HOSTNAME) → prompt for login (if needed) and run a named tunnel to that hostname.

LOCAL_PORT=6080
HOSTNAME="${TUNNEL_HOSTNAME:-${1:-}}"
TUNNEL_NAME="${TUNNEL_NAME:-${2:-myapp}}"
CF_DIR="$HOME/.cloudflared"

if [[ -z "$HOSTNAME" ]]; then
  echo "[tunnel] Target: http://localhost:${LOCAL_PORT} (ephemeral)"
else
  echo "[tunnel] Target: https://${HOSTNAME} → http://localhost:${LOCAL_PORT} (named tunnel: ${TUNNEL_NAME})"
fi

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

if [[ -z "$HOSTNAME" ]]; then
  # Ephemeral (trycloudflare) mode
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
else
  # Named tunnel mode
  if [[ ! -f "$CF_DIR/cert.pem" ]]; then
    echo "[tunnel] Login required. A browser will open; complete Cloudflare auth..."
    cloudflared tunnel login
  fi

  # Create tunnel if missing
  credentials_file=""
  if cloudflared tunnel info "$TUNNEL_NAME" >/dev/null 2>&1; then
    credentials_file=$(ls -t "$CF_DIR"/*.json 2>/dev/null | head -n1 || true)
  else
    echo "[tunnel] Creating tunnel '$TUNNEL_NAME'..."
    create_out=$(cloudflared tunnel create "$TUNNEL_NAME" 2>&1 | tee /dev/stderr)
    credentials_file=$(echo "$create_out" | grep -oE "$CF_DIR/[a-f0-9-]+\.json" | head -n1 || true)
    if [[ -z "$credentials_file" ]]; then
      credentials_file=$(ls -t "$CF_DIR"/*.json 2>/dev/null | head -n1 || true)
    fi
  fi

  if [[ -z "$credentials_file" ]]; then
    echo "[tunnel] ERROR: Could not find tunnel credentials in $CF_DIR" >&2
    exit 1
  fi

  # Write config mapping hostname → localhost:6080
  cat > "$CF_DIR/config.yml" <<EOF
tunnel: $TUNNEL_NAME
credentials-file: $credentials_file
ingress:
  - hostname: $HOSTNAME
    service: http://localhost:6080
  - service: http_status:404
EOF

  # Route DNS (creates proxied CNAME)
  cloudflared tunnel route dns "$TUNNEL_NAME" "$HOSTNAME" || true

  echo "[tunnel] Running tunnel '$TUNNEL_NAME' for https://$HOSTNAME → http://localhost:6080"
  exec cloudflared tunnel run "$TUNNEL_NAME"
fi
