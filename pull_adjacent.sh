#!/bin/bash
set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)"
parent="$(dirname "$src")"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

repos=()
pids=()

pull_repo() {
    local dir="$1" log="$2"
    if git -C "$dir" pull --ff-only >"$log" 2>&1; then
        :
    else
        echo "  ⚠ pull failed for $dir" >>"$log"
    fi
}

repos+=("$src")
pull_repo "$src" "$tmpdir/0.log" &
pids+=($!)

i=1
for dir in "$parent"/unity_*/; do
    [ "$dir" = "$src/" ] && continue
    [ -d "$dir/.git" ] || continue
    repos+=("$dir")
    pull_repo "$dir" "$tmpdir/$i.log" &
    pids+=($!)
    ((i++))
done

for i in "${!repos[@]}"; do
    wait "${pids[$i]}"
    echo "Pulling: ${repos[$i]}"
    cat "$tmpdir/$i.log"
done
