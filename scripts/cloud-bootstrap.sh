#!/usr/bin/env bash
# Cloud bootstrap script for Cursor Cloud Agents.
#
# This script clones the required sibling repositories (unify, unillm) before
# running uv sync. The pyproject.toml references these as editable installs at
# ../unify and ../unillm, which exist locally but not in cloud environments.
#
# Branch selection follows the same logic as GitHub Actions:
# - main branch → clone main branches
# - other branches → clone staging branches

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Determine current unity branch
UNITY_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "staging")

# Select branch for sibling repos (main→main, otherwise→staging)
if [[ "$UNITY_BRANCH" == "main" ]]; then
    SIBLING_BRANCH="main"
else
    SIBLING_BRANCH="staging"
fi

echo "=== Cloud Bootstrap ==="
echo "Unity branch: $UNITY_BRANCH"
echo "Sibling repos branch: $SIBLING_BRANCH"

# Debug: Check which auth tokens are available
echo ""
echo "Auth tokens available:"
[[ -n "${CLONE_TOKEN:-}" ]] && echo "  ✓ CLONE_TOKEN is set" || echo "  ✗ CLONE_TOKEN not set"
[[ -n "${GH_TOKEN:-}" ]] && echo "  ✓ GH_TOKEN is set" || echo "  ✗ GH_TOKEN not set"
[[ -n "${GITHUB_TOKEN:-}" ]] && echo "  ✓ GITHUB_TOKEN is set" || echo "  ✗ GITHUB_TOKEN not set"
command -v gh >/dev/null 2>&1 && echo "  ✓ gh CLI available" || echo "  ✗ gh CLI not available"
echo ""

# Clone sibling repos to parent directory (so ../unify and ../unillm work)
PARENT_DIR="$(dirname "$REPO_ROOT")"

clone_repo() {
    local repo="$1"
    local target="$PARENT_DIR/$repo"

    if [[ -d "$target" ]]; then
        echo "✓ $repo already exists at $target"
        return 0
    fi

    echo "Cloning $repo to $target..."

    # Try CLONE_TOKEN (org-wide cross-repo token)
    if [[ -n "${CLONE_TOKEN:-}" ]]; then
        echo "  Trying CLONE_TOKEN..."
        if git clone --branch "$SIBLING_BRANCH" --depth 1 \
            "https://x-access-token:${CLONE_TOKEN}@github.com/unifyai/$repo.git" "$target"; then
            echo "✓ Cloned $repo via CLONE_TOKEN"
            return 0
        fi
        echo "  CLONE_TOKEN clone failed"
    fi

    # Try gh CLI (uses Cursor's GitHub auth)
    if command -v gh >/dev/null 2>&1; then
        echo "  Trying gh CLI..."
        if gh repo clone "unifyai/$repo" "$target" -- --branch "$SIBLING_BRANCH" --depth 1; then
            echo "✓ Cloned $repo via gh CLI"
            return 0
        fi
        echo "  gh CLI clone failed"
    fi

    # Fall back to generic GH_TOKEN/GITHUB_TOKEN
    if [[ -n "${GH_TOKEN:-}" ]] || [[ -n "${GITHUB_TOKEN:-}" ]]; then
        local token="${GH_TOKEN:-$GITHUB_TOKEN}"
        echo "  Trying GH_TOKEN/GITHUB_TOKEN..."
        if git clone --branch "$SIBLING_BRANCH" --depth 1 \
            "https://x-access-token:${token}@github.com/unifyai/$repo.git" "$target"; then
            echo "✓ Cloned $repo via git with token"
            return 0
        fi
        echo "  GH_TOKEN/GITHUB_TOKEN clone failed"
    fi

    # Try unauthenticated (works for public repos)
    echo "  Trying unauthenticated..."
    if git clone --branch "$SIBLING_BRANCH" --depth 1 \
        "https://github.com/unifyai/$repo.git" "$target"; then
        echo "✓ Cloned $repo (public)"
        return 0
    fi

    echo "ERROR: Failed to clone $repo"
    echo "Add CLONE_TOKEN to Cursor Cloud environment secrets."
    echo "(Use the org-wide cross-repo token from GitHub Actions secrets)"
    return 1
}

# Clone required sibling repos
clone_repo "unify"
clone_repo "unillm"

echo ""
echo "=== Installing dependencies ==="
uv sync --all-groups

echo ""
echo "✓ Cloud bootstrap complete"
