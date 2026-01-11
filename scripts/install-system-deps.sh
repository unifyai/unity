#!/usr/bin/env bash
# Shared system dependencies for both production Dockerfile and Cursor Cloud Agent.
# This is the SINGLE SOURCE OF TRUTH for apt packages needed by the project.
#
# Usage (in Dockerfiles):
#   COPY scripts/install-system-deps.sh /tmp/
#   RUN /tmp/install-system-deps.sh [--minimal]
#
# Flags:
#   --minimal   Only install deps needed for testing (skips X11, browsers, VNC, etc.)

set -euo pipefail

MINIMAL=false
for arg in "$@"; do
    case "$arg" in
        --minimal) MINIMAL=true ;;
    esac
done

export DEBIAN_FRONTEND=noninteractive

apt-get update

# =============================================================================
# CORE BUILD DEPENDENCIES (always needed)
# =============================================================================
apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    pkg-config \
    python3-dev \
    git \
    curl \
    sudo

# =============================================================================
# GITHUB CLI (for downloading CI artifacts and GitHub operations)
# =============================================================================
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
apt-get update
apt-get install -y --no-install-recommends gh

# =============================================================================
# AUDIO LIBRARIES (needed for pyaudio)
# =============================================================================
apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libasound2 \
    libasound2-plugins \
    libportaudio2 \
    libpulse0

# =============================================================================
# REDIS (for caching)
# =============================================================================
apt-get install -y --no-install-recommends redis-server

# =============================================================================
# TMUX (for parallel test runner)
# =============================================================================
apt-get install -y --no-install-recommends tmux

# =============================================================================
# ZSH (for shell function testing)
# =============================================================================
apt-get install -y --no-install-recommends zsh

# =============================================================================
# LOCALES (for UTF-8 emoji support)
# =============================================================================
apt-get install -y --no-install-recommends locales
sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen

# =============================================================================
# POWERSHELL CORE (for shell function testing)
# =============================================================================
# Detect distro to use correct Microsoft repository
if [ -f /etc/os-release ]; then
    . /etc/os-release
    case "$ID-$VERSION_CODENAME" in
        ubuntu-jammy)   MSFT_REPO="microsoft-ubuntu-jammy-prod jammy" ;;
        ubuntu-noble)   MSFT_REPO="microsoft-ubuntu-noble-prod noble" ;;
        debian-bookworm) MSFT_REPO="microsoft-debian-bookworm-prod bookworm" ;;
        debian-bullseye) MSFT_REPO="microsoft-debian-bullseye-prod bullseye" ;;
        *)
            echo "Warning: Unknown distro $ID-$VERSION_CODENAME, skipping PowerShell installation"
            MSFT_REPO=""
            ;;
    esac
else
    echo "Warning: Cannot detect distro, skipping PowerShell installation"
    MSFT_REPO=""
fi

if [ -n "$MSFT_REPO" ]; then
    curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/$MSFT_REPO main" | tee /etc/apt/sources.list.d/microsoft.list > /dev/null
    apt-get update
    apt-get install -y --no-install-recommends powershell
    echo "PowerShell installed: $(pwsh --version)"
fi

# =============================================================================
# FULL PRODUCTION DEPENDENCIES (skipped with --minimal)
# =============================================================================
if [ "$MINIMAL" = false ]; then
    # Process manager
    apt-get install -y --no-install-recommends tini

    # Download utilities
    apt-get install -y --no-install-recommends \
        wget \
        unzip \
        gnupg2 \
        ca-certificates

    # X11 / Virtual desktop / VNC
    apt-get install -y --no-install-recommends \
        xvfb \
        x11vnc \
        fluxbox \
        xdotool \
        wmctrl \
        xterm \
        dbus \
        dbus-x11 \
        websockify

    # XDG desktop portals
    apt-get install -y --no-install-recommends \
        xdg-desktop-portal \
        xdg-desktop-portal-gtk

    # Browser runtime dependencies
    apt-get install -y --no-install-recommends \
        libnss3 \
        libatk-bridge2.0-0 \
        libgtk-3-0 \
        libxss1 \
        libxshmfence1 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libgbm1 \
        libx11-xcb1 \
        fonts-liberation \
        xdg-utils

    # Media processing
    apt-get install -y --no-install-recommends ffmpeg

    # Full audio stack (PipeWire)
    apt-get install -y --no-install-recommends \
        pipewire \
        pipewire-pulse \
        pipewire-alsa \
        wireplumber \
        pulseaudio-utils \
        alsa-utils

    # Filesystem utilities (for AppImage support)
    apt-get install -y --no-install-recommends \
        fuse3 \
        libfuse2 \
        squashfs-tools

    # Image processing
    apt-get install -y --no-install-recommends libvips

    # GTK4 and additional browser dependencies
    apt-get install -y --no-install-recommends \
        libgtk-4-1 \
        libharfbuzz-icu0 \
        libenchant-2-2 \
        libsecret-1-0 \
        libhyphen0 \
        libmanette-0.2-0
fi

# Cleanup apt cache
rm -rf /var/lib/apt/lists/*

echo "System dependencies installed successfully (minimal=$MINIMAL)."
