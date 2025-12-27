#!/usr/bin/env bash
# Auto-configure git identity from repo history.
# Used by Cursor Cloud Agent to avoid commits with "cursoragent" identity.
#
# Logic:
# 1. Check if git user is already configured (non-cursoragent) - if so, skip
# 2. Otherwise, detect the most recent human author from git log
# 3. Configure git with that identity

set -e

# Check if we're in a git repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "[configure-git-identity] Not a git repository, skipping"
    exit 0
fi

# Get current git config
CURRENT_NAME=$(git config --global user.name 2>/dev/null || echo "")
CURRENT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

# If already configured with a non-cursoragent identity, we're done
if [[ -n "$CURRENT_NAME" && -n "$CURRENT_EMAIL" && "$CURRENT_EMAIL" != *"cursoragent"* ]]; then
    echo "[configure-git-identity] Git identity already configured: $CURRENT_NAME <$CURRENT_EMAIL>"
    exit 0
fi

# Find the most recent non-cursoragent author from git log
# Look at the last 50 commits to find a human author
AUTHOR_INFO=$(git log --format="%an|%ae" -50 2>/dev/null | grep -v "cursoragent" | head -1 || echo "")

if [[ -z "$AUTHOR_INFO" ]]; then
    echo "[configure-git-identity] WARNING: Could not detect author from git history"
    echo "[configure-git-identity] Please configure manually: git config --global user.name 'Your Name' && git config --global user.email 'you@example.com'"
    exit 0
fi

# Parse author name and email
DETECTED_NAME=$(echo "$AUTHOR_INFO" | cut -d'|' -f1)
DETECTED_EMAIL=$(echo "$AUTHOR_INFO" | cut -d'|' -f2)

if [[ -n "$DETECTED_NAME" && -n "$DETECTED_EMAIL" ]]; then
    git config --global user.name "$DETECTED_NAME"
    git config --global user.email "$DETECTED_EMAIL"
    echo "[configure-git-identity] Configured git identity: $DETECTED_NAME <$DETECTED_EMAIL>"
else
    echo "[configure-git-identity] WARNING: Could not parse author info"
fi
