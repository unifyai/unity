#!/usr/bin/env bash
# Install git hooks that prevent commits with cursoragent identity.
# This provides a symbolic guarantee - such commits simply cannot be made.

set -e

# Check if we're in a git repo
GIT_DIR=$(git rev-parse --git-dir 2>/dev/null) || {
    echo "[install-git-hooks] Not a git repository, skipping"
    exit 0
}

HOOKS_DIR="$GIT_DIR/hooks"
mkdir -p "$HOOKS_DIR"

# Create pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'HOOK_EOF'
#!/usr/bin/env bash
# Pre-commit hook: reject commits with cursoragent identity
# This provides a hard symbolic guarantee against cursoragent commits.

# Get the author/committer that will be used for this commit
AUTHOR_NAME=$(git config user.name 2>/dev/null || echo "")
AUTHOR_EMAIL=$(git config user.email 2>/dev/null || echo "")

# Check for cursoragent in author
if [[ "$AUTHOR_NAME" == *"cursoragent"* || "$AUTHOR_NAME" == *"Cursor Agent"* ]]; then
    echo "ERROR: Commit blocked - author name contains 'cursoragent'"
    echo ""
    echo "Please configure your git identity:"
    echo "  git config --global user.name 'Your Name'"
    echo "  git config --global user.email 'you@example.com'"
    echo ""
    echo "Or run: ./scripts/configure-git-identity.sh"
    exit 1
fi

if [[ "$AUTHOR_EMAIL" == *"cursoragent"* ]]; then
    echo "ERROR: Commit blocked - author email contains 'cursoragent'"
    echo ""
    echo "Please configure your git identity:"
    echo "  git config --global user.name 'Your Name'"
    echo "  git config --global user.email 'you@example.com'"
    echo ""
    echo "Or run: ./scripts/configure-git-identity.sh"
    exit 1
fi

# Also check GIT_AUTHOR_* and GIT_COMMITTER_* env vars if set
if [[ "${GIT_AUTHOR_NAME:-}" == *"cursoragent"* || "${GIT_AUTHOR_NAME:-}" == *"Cursor Agent"* ]]; then
    echo "ERROR: Commit blocked - GIT_AUTHOR_NAME contains 'cursoragent'"
    exit 1
fi

if [[ "${GIT_AUTHOR_EMAIL:-}" == *"cursoragent"* ]]; then
    echo "ERROR: Commit blocked - GIT_AUTHOR_EMAIL contains 'cursoragent'"
    exit 1
fi

if [[ "${GIT_COMMITTER_NAME:-}" == *"cursoragent"* || "${GIT_COMMITTER_NAME:-}" == *"Cursor Agent"* ]]; then
    echo "ERROR: Commit blocked - GIT_COMMITTER_NAME contains 'cursoragent'"
    exit 1
fi

if [[ "${GIT_COMMITTER_EMAIL:-}" == *"cursoragent"* ]]; then
    echo "ERROR: Commit blocked - GIT_COMMITTER_EMAIL contains 'cursoragent'"
    exit 1
fi

exit 0
HOOK_EOF

chmod +x "$HOOKS_DIR/pre-commit"
echo "[install-git-hooks] Installed pre-commit hook to block cursoragent commits"
