#!/bin/bash
set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)"
parent="$(dirname "$src")"

echo "Pulling: $src"
git -C "$src" pull --ff-only || echo "  ⚠ pull failed for $src"

for dir in "$parent"/unity_*/; do
    [ "$dir" = "$src/" ] && continue
    [ -d "$dir/.git" ] || continue
    echo "Pulling: $dir"
    git -C "$dir" pull --ff-only || echo "  ⚠ pull failed for $dir"
done
