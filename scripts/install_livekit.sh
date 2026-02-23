#!/usr/bin/env bash
#
# Install livekit-server binary for end-to-end call tests.
#
# Usage:
#   ./scripts/install_livekit.sh           # Install to /usr/local/bin (requires sudo)
#   ./scripts/install_livekit.sh ~/.local/bin  # Install to custom directory
#
# This script:
# 1. Detects OS and architecture
# 2. Downloads the appropriate binary from GitHub releases
# 3. Installs it to the specified directory
#
# After installation, run tests with:
#   ./tests/parallel_run.sh tests/conversation_manager/voice/test_e2e_call_flow.py

set -euo pipefail

# Default version and install directory
LIVEKIT_VERSION="${LIVEKIT_VERSION:-1.8.4}"
INSTALL_DIR="${1:-/usr/local/bin}"

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux)  OS_NAME="linux" ;;
    Darwin) OS_NAME="darwin" ;;
    *)      echo "Unsupported OS: $OS"; exit 1 ;;
esac

# Detect architecture
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64) ARCH_NAME="amd64" ;;
    arm64|aarch64) ARCH_NAME="arm64" ;;
    *)      echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

# Check if already installed
if command -v livekit-server &>/dev/null; then
    INSTALLED_VERSION=$(livekit-server --version 2>&1 | head -1 || echo "unknown")
    echo "livekit-server already installed: $INSTALLED_VERSION"
    echo "To reinstall, remove it first: rm $(which livekit-server)"
    exit 0
fi

# Construct download URL
FILENAME="livekit_${LIVEKIT_VERSION}_${OS_NAME}_${ARCH_NAME}.tar.gz"
URL="https://github.com/livekit/livekit/releases/download/v${LIVEKIT_VERSION}/${FILENAME}"

echo "Installing livekit-server v${LIVEKIT_VERSION} for ${OS_NAME}/${ARCH_NAME}..."
echo "Download URL: $URL"
echo "Install directory: $INSTALL_DIR"

# Create temp directory
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# Download and extract
echo "Downloading..."
curl -fsSL "$URL" -o "$TMP_DIR/livekit.tar.gz"

echo "Extracting..."
tar -xzf "$TMP_DIR/livekit.tar.gz" -C "$TMP_DIR"

# Install binary
echo "Installing to $INSTALL_DIR..."
if [[ -w "$INSTALL_DIR" ]]; then
    mv "$TMP_DIR/livekit-server" "$INSTALL_DIR/"
else
    sudo mv "$TMP_DIR/livekit-server" "$INSTALL_DIR/"
fi

chmod +x "$INSTALL_DIR/livekit-server"

# Verify installation
if command -v livekit-server &>/dev/null; then
    echo ""
    echo "✅ livekit-server installed successfully!"
    echo "   Version: $(livekit-server --version 2>&1 | head -1)"
    echo ""
    echo "To run LiveKit in dev mode:"
    echo "   livekit-server --dev"
    echo ""
    echo "To run the e2e tests:"
    echo "   ./tests/parallel_run.sh tests/conversation_manager/voice/test_e2e_call_flow.py"
else
    echo ""
    echo "⚠️  livekit-server installed but not in PATH"
    echo "   Add $INSTALL_DIR to your PATH, or specify full path"
fi
