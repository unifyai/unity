#!/bin/bash
set -euo pipefail

purpose="${1:?Usage: $0 <purpose>}"
src="$(cd "$(dirname "$0")" && pwd)"
dst="$src/../unity_${purpose}"

[ -d "$dst" ] && { echo "Already exists: $dst" >&2; exit 1; }

origin=$(git -C "$src" remote get-url origin)
git clone --branch staging "$origin" "$dst"

cd "$dst"
git submodule update --init --recursive
cp "$src/.env" .env
ln -sf "$src/.venv" .venv

.venv/bin/python -m pre_commit install

echo "Ready: $dst"
