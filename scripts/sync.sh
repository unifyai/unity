#!/bin/bash
# Sync all dependencies, including latest commits from git-based dependencies.
#
# Usage: ./scripts/sync.sh
#
# This script upgrades git dependencies (unifyai, unillm) to their latest
# commits before running uv sync. Use this instead of plain `uv sync` when
# you want to pull the latest from upstream git repos.

set -e

cd "$(dirname "$0")/.."

echo "Upgrading git dependencies to latest commits..."
uv lock --upgrade-package unifyai --upgrade-package unillm

echo "Syncing all dependencies..."
uv sync --all-groups

echo "Done."
