#!/usr/bin/env bash
set -euo pipefail

# Check and install cloudflared if missing
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "[tunnel] cloudflared not found. Installing..."
  # Add cloudflare GPG key and apt source
  mkdir -p --mode=0755 /usr/share/keyrings
  curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
  echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' | tee /etc/apt/sources.list.d/cloudflared.list >/dev/null
  apt-get update
  apt-get install -y cloudflared
  echo "[tunnel] cloudflared installed."
else
  echo "[tunnel] cloudflared is already installed: $(cloudflared --version | head -n1)"
fi

# Configure a named tunnel to forward localhost:3000 to a custom domain
# Usage: TUNNEL_HOSTNAME=myapp.example.com TUNNEL_NAME=myapp bash tunnel.sh

HOSTNAME="${TUNNEL_HOSTNAME:-${1:-}}"
TUNNEL_NAME="${TUNNEL_NAME:-${2:-myapp}}"
CF_DIR="$HOME/.cloudflared"

if [ -z "$HOSTNAME" ]; then
  echo "[tunnel] INFO: No hostname provided. Starting ad-hoc tunnel for testing..."
  exec cloudflared tunnel --url http://localhost:3000
fi

if [ ! -f "$CF_DIR/cert.pem" ]; then
  echo "[tunnel] ERROR: cloudflared is not logged in. Run: cloudflared tunnel login" >&2
  exit 1
fi

# Create tunnel if missing
credentials_file=""
if cloudflared tunnel info "$TUNNEL_NAME" >/dev/null 2>&1; then
  credentials_file=$(ls -t "$CF_DIR"/*.json 2>/dev/null | head -n1 || true)
else
  echo "[tunnel] Creating tunnel '$TUNNEL_NAME'..."
  create_out=$(cloudflared tunnel create "$TUNNEL_NAME" 2>&1 | tee /dev/stderr)
  credentials_file=$(echo "$create_out" | grep -oE "$CF_DIR/[a-f0-9-]+\.json" | head -n1 || true)
  if [ -z "$credentials_file" ]; then
    credentials_file=$(ls -t "$CF_DIR"/*.json 2>/dev/null | head -n1 || true)
  fi
fi

if [ -z "$credentials_file" ]; then
  echo "[tunnel] ERROR: Could not find tunnel credentials in $CF_DIR" >&2
  exit 1
fi

# Write config mapping hostname → localhost:3000
cat > "$CF_DIR/config.yml" <<EOF
tunnel: $TUNNEL_NAME
credentials-file: $credentials_file
ingress:
  - hostname: $HOSTNAME
    service: http://localhost:3000
  - service: http_status:404
EOF

# Route DNS (creates proxied CNAME)
cloudflared tunnel route dns "$TUNNEL_NAME" "$HOSTNAME" || true

echo "[tunnel] Running tunnel '$TUNNEL_NAME' for https://$HOSTNAME → http://localhost:3000"
exec cloudflared tunnel run "$TUNNEL_NAME"
