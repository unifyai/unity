#!/usr/bin/env bash
# Configure git identity for Cursor Cloud Agents.
#
# Priority order:
# 1. If GIT_USER_NAME and GIT_USER_EMAIL env vars are set, use them (recommended)
# 2. If git user is already configured (non-cursoragent), keep it
# 3. Otherwise, detect the most recent human author from git log (fallback)
#
# For Cloud Agents: Set GIT_USER_NAME and GIT_USER_EMAIL as user-specific secrets
# in Cursor Settings → Cloud Agents → Secrets.

set -e

# Check if we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "[configure-git-identity] Not a git repository, skipping"
    exit 0
fi

# Priority 1: Use explicit environment variables (recommended for Cloud Agents)
if [[ -n "$GIT_USER_NAME" && -n "$GIT_USER_EMAIL" ]]; then
    git config --global user.name "$GIT_USER_NAME"
    git config --global user.email "$GIT_USER_EMAIL"
    echo "[configure-git-identity] Configured from env vars: $GIT_USER_NAME <$GIT_USER_EMAIL>"
    exit 0
fi

# Get current git config
CURRENT_NAME=$(git config --global user.name 2>/dev/null || echo "")
CURRENT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

# Priority 2: If already configured with a non-cursoragent identity, we're done
if [[ -n "$CURRENT_NAME" && -n "$CURRENT_EMAIL" && "$CURRENT_EMAIL" != *"cursoragent"* ]]; then
    echo "[configure-git-identity] Git identity already configured: $CURRENT_NAME <$CURRENT_EMAIL>"
    exit 0
fi

# Priority 3: Fallback - detect from git history
# Look at the last 50 commits to find a human author
AUTHOR_INFO=$(git log --format="%an|%ae" -50 2>/dev/null | grep -v "cursoragent" | head -1 || echo "")

if [[ -z "$AUTHOR_INFO" ]]; then
    echo "[configure-git-identity] WARNING: Could not detect author from git history"
    echo "[configure-git-identity] Please set GIT_USER_NAME and GIT_USER_EMAIL in Cursor Settings → Cloud Agents → Secrets"
    exit 0
fi

# Parse author name and email
DETECTED_NAME=$(echo "$AUTHOR_INFO" | cut -d'|' -f1)
DETECTED_EMAIL=$(echo "$AUTHOR_INFO" | cut -d'|' -f2)

if [[ -n "$DETECTED_NAME" && -n "$DETECTED_EMAIL" ]]; then
    git config --global user.name "$DETECTED_NAME"
    git config --global user.email "$DETECTED_EMAIL"
    echo "[configure-git-identity] Configured from git history: $DETECTED_NAME <$DETECTED_EMAIL>"
else
    echo "[configure-git-identity] WARNING: Could not parse author info"
fi
