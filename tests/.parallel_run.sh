#!/usr/bin/env bash
set -euo pipefail

# Optionally source environment from ../.env (relative to tests directory)
# This allows storing secrets/config like UNIFY_KEY in ~/unity/.env
if [ -f "../.env" ]; then
  # shellcheck disable=SC1091
  set -a
  . "../.env"
  set +a
fi

# ---- Configurable directory excludes (by name) ----
EXCLUDE_DIRS=( .git .hg .svn .venv venv .mypy_cache .pytest_cache __pycache__ .idea .vscode )

# ---- Modes ----
# Default: one session per file.
# With -t/--per-test: one session per collected pytest node id across provided dirs/files.
PER_TEST=0

# Resolve repo root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

# Parse flags; collect positional args
declare -a POSITIONAL_ARGS=()
for _arg in "$@"; do
  case "$_arg" in
    -t|--per-test) PER_TEST=1 ;;
    *) POSITIONAL_ARGS+=( "$_arg" ) ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]}"

# Build the command to run in each tmux session
run_cmd() {
  local target="$1"   # pytest target (file path or node id)
  # Run pytest; then change ONLY the leading status prefix on the current tmux session:
  printf "bash -lc 'export UNIFY_TESTS_RAND_PROJ=True UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True; source ~/unity/.venv/bin/activate && cd %q && pytest %q; status=\$?; sname=\$(tmux display-message -p -t \"\$TMUX_PANE\" \"#{session_name}\"); base=\"\$sname\"; case \"\$sname\" in \"o ✅ \"*) base=\"\${sname#o ✅ }\" ;; \"x ❌ \"*) base=\"\${sname#x ❌ }\" ;; \"? ⏳ \"*) base=\"\${sname#? ⏳ }\" ;; \"✅ \"*) base=\"\${sname#✅ }\" ;; \"❌ \"*) base=\"\${sname#❌ }\" ;; \"⏳ \"*) base=\"\${sname#⏳ }\" ;; esac; if [ \$status -eq 0 ]; then pfx=\"o ✅\"; else pfx=\"x ❌\"; fi; tmux rename-session -t \"\$sname\" \"\$pfx \$base\"; if [ \$status -eq 0 ]; then sid=\$(tmux display-message -p -t \"\$TMUX_PANE\" \"#{session_id}\"); (sleep 10; tmux kill-session -t \"\$sid\") >/dev/null 2>&1 & disown; echo \"All tests passed. This tmux session will close in 10s...\"; fi; echo; echo \"pytest exited with code: \$status\"; echo \"(You are now in a shell. Press Ctrl-D to close this window.)\"; exec bash -l'" "$REPO_ROOT" "$target"
}

# Ensure we don't collide with existing sessions
unique_session_name() {
  local base="$1" name="$1" n=1
  while tmux has-session -t "$name" 2>/dev/null; do
    ((n++)); name="${base}-${n}"
  done
  printf "%s" "$name"
}

# Turn a file path (or pytest node id) into a session base name
#   ./animals/dogs/test_bark.py               -> animals-dogs-test_bark
#   ./animals/dogs/test_bark.py::test_woof    -> animals-dogs-test_bark--test_woof
session_basename_for() {
  local original="$1"
  local p
  local node_suffix=""

  # If a pytest node id is provided, split off the suffix after "::"
  if [[ "$original" == *"::"* ]]; then
    local base="${original%%::*}"
    node_suffix="${original#${base}::}"
    p="$base"
  else
    p="$original"
  fi

  # normalize to a relative-looking path for naming
  [[ "$p" = /* ]] || p="./${p#./}"
  p="${p%.py}"
  p="${p#./}"
  # Drop leading 'tests/' to avoid 'tests-' prefix in session names
  p="${p#tests/}"
  p="${p//\//-}"

  # If we have a node suffix, sanitize it and append
  if [[ -n "$node_suffix" ]]; then
    local ns="$node_suffix"
    ns="${ns//::/-}"
    ns="${ns// /-}"
    ns="${ns//[/}"
    ns="${ns//]/}"
    ns="${ns//(/}"
    ns="${ns//)/}"
    ns="${ns//,/}"
    ns="${ns//:/-}"
    ns="${ns//=/-}"
    ns="${ns//./-}"
    p="${p}--${ns}"
  fi

  printf "%s" "$p"
}

# Collect args: files and/or directories to search
declare -a roots=()
declare -a direct_files=()
declare -a direct_nodes=()

if (( $# == 0 )); then
  roots=( "." )
else
  for arg in "$@"; do
    if [[ -f "$arg" ]]; then
      # only include Python test files directly (names starting with test_)
      if [[ "${arg##*/}" == test_*.py ]]; then
        direct_files+=( "$arg" )
      fi
    elif [[ "$arg" == *"::"* ]]; then
      # pytest node id: extract base file and validate it exists and is a test file
      base="${arg%%::*}"
      if [[ -f "$base" ]]; then
        if [[ "${base##*/}" == test_*.py ]]; then
          direct_nodes+=( "$arg" )
        else
          echo "Warning: Skipping node not under a test_*.py file: $arg" >&2
        fi
      else
        echo "Warning: Skipping non-existent test node (file missing): $arg" >&2
      fi
    elif [[ -d "$arg" ]]; then
      roots+=( "$arg" )
    else
      echo "Warning: Skipping non-existent path: $arg" >&2
    fi
  done
  if (( ${#roots[@]} == 0 && ${#direct_files[@]} == 0 && ${#direct_nodes[@]} == 0 )); then
    echo "No valid directories, files, or tests provided." >&2
    exit 1
  fi
fi

# Build a safe find pipeline:
# find <roots> \( -type d \( -name EX1 -o EX2 ... \) -prune \) -o \( -type f -name "test_*.py" -print0 \)
build_find_cmd() {
  local -a cmd=( find )
  if (( ${#roots[@]} )); then
    cmd+=( "${roots[@]}" )
  else
    cmd+=( "." )
  fi

  cmd+=( "(" -type d "(" )
  local first=1
  for d in "${EXCLUDE_DIRS[@]}"; do
    if (( first )); then
      cmd+=( -name "$d" )
      first=0
    else
      cmd+=( -o -name "$d" )
    fi
  done
  cmd+=( ")" -prune ")" -o "(" -type f -name "test_*.py" -print0 ")" )

  printf '%q ' "${cmd[@]}"
}

# Collect pytest node ids for a given target (file or directory)
collect_nodes_for_target() {
  local target="$1"
  local cmd
  cmd=$(printf 'export UNIFY_TESTS_RAND_PROJ=True UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True; source ~/unity/.venv/bin/activate && pytest --collect-only -q %q' "$target")
  # Remove color codes, keep only node ids (contain ::), ignore noise; never fail the script
  bash -lc "$cmd" 2>/dev/null | sed -E 's/\x1B\[[0-9;]*[mK]//g' | grep -E '::' || true
}

# Gather recursive .py files from roots (NUL-delimited, sorted)
declare -a found_files=()
if (( ${#roots[@]} )); then
  found_files=()
  while IFS= read -r -d '' f; do
    found_files+=( "$f" )
  done < <(eval "$(build_find_cmd)")
fi

# Combine targets based on mode; sort deterministically (and de-duplicate)
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
if (( PER_TEST )); then
  # Per-test mode: expand directories/files into node ids
  if (( ${#direct_files[@]} )); then
    for f in "${direct_files[@]}"; do
      while IFS= read -r nid; do
        [[ -n "$nid" ]] && printf '%s\0' "$nid" >> "$tmp"
      done < <(collect_nodes_for_target "$f")
    done
  fi
  if (( ${#found_files[@]} )); then
    for f in "${found_files[@]}"; do
      while IFS= read -r nid; do
        [[ -n "$nid" ]] && printf '%s\0' "$nid" >> "$tmp"
      done < <(collect_nodes_for_target "$f")
    done
  fi
  if (( ${#direct_nodes[@]} )); then
    printf '%s\0' "${direct_nodes[@]}" >> "$tmp"
  fi
else
  # Default mode: one session per file; explicit node ids are respected
  if (( ${#direct_files[@]} )); then
    printf '%s\0' "${direct_files[@]}" >> "$tmp"
  fi
  if (( ${#found_files[@]} )); then
    printf '%s\0' "${found_files[@]}" >> "$tmp"
  fi
  if (( ${#direct_nodes[@]} )); then
    printf '%s\0' "${direct_nodes[@]}" >> "$tmp"
  fi
fi

files=()
while IFS= read -r -d '' f; do
  files+=( "$f" )
done < <(tr '\0' '\n' < "$tmp" | LC_ALL=C sort -u | tr '\n' '\0')

if (( ${#files[@]} == 0 )); then
  echo "No tests found."
  exit 0
fi

declare -a made_sessions=()
for target in "${files[@]}"; do
  base_sess="$(session_basename_for "$target")"
  session="$(unique_session_name "$base_sess")"

  # Window name = basename without .py
  fname="${target##*/}"
  wname="${fname%.py}"

  # Create the session first (no command), set remain-on-exit, then send the command.
  tmux new-session -d -s "$session" -n "$wname"
  pending_name="$(unique_session_name "? ⏳ $session")"
  tmux rename-session -t "$session" "$pending_name"
  session="$pending_name"
  tmux send-keys -t "$session:" "$(run_cmd "$target")" C-m

  made_sessions+=( "$session" )
done

echo "Created ${#made_sessions[@]} tmux sessions:"
for s in "${made_sessions[@]}"; do
  echo "  - $s"
done

echo
echo "Trigger:"
echo "  • Run everything under current dir:     ./\\.parallel_run.sh"
echo "  • Only a folder:                         ./\\.parallel_run.sh test_cats"
echo "  • Multiple roots:                        ./\\.parallel_run.sh tests/unit tests/integration"
echo "  • Specific files:                        ./\\.parallel_run.sh tests/test_foo.py tests/test_bar.py"
echo "  • Specific tests:                        ./\\.parallel_run.sh tests/test_foo.py::TestA::test_x tests/test_bar.py::test_y"
echo "  • Per-test (dirs/files):                 ./\\.parallel_run.sh -t tests tests/test_foo.py"
echo "  • Per-test (everything here):            ./\\.parallel_run.sh -t"
echo
echo "Observe:"
echo "  • Watch sessions: watch -n 0.5 'tmux ls'"
echo "  • List sessions: tmux ls"
echo "  • Attach:       tmux attach -t <session>"
echo "  • Inside tmux:  tmux switch-client -t <session>"
