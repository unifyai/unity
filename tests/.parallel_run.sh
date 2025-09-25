#!/usr/bin/env bash
set -euo pipefail

# ---- Configurable directory excludes (by name) ----
EXCLUDE_DIRS=( .git .hg .svn .venv venv .mypy_cache .pytest_cache __pycache__ .idea .vscode )

# Build the command to run in each tmux session
run_cmd() {
  local target="$1"   # file path (relative or absolute)
  # Run pytest; then change ONLY the leading status prefix on the current tmux session:
  printf "bash -lc 'export UNIFY_TESTS_RAND_PROJ=True UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True; source ~/unity/.unity/bin/activate && pytest %q; status=\$?; sname=\$(tmux display-message -p -t \"\$TMUX_PANE\" \"#{session_name}\"); base=\"\$sname\"; case \"\$sname\" in \"o ✅ \"*) base=\"\${sname#o ✅ }\" ;; \"x ❌ \"*) base=\"\${sname#x ❌ }\" ;; \"? ⏳ \"*) base=\"\${sname#? ⏳ }\" ;; \"✅ \"*) base=\"\${sname#✅ }\" ;; \"❌ \"*) base=\"\${sname#❌ }\" ;; \"⏳ \"*) base=\"\${sname#⏳ }\" ;; esac; if [ \$status -eq 0 ]; then pfx=\"o ✅\"; else pfx=\"x ❌\"; fi; tmux rename-session -t \"\$sname\" \"\$pfx \$base\"; if [ \$status -eq 0 ]; then sid=\$(tmux display-message -p -t \"\$TMUX_PANE\" \"#{session_id}\"); (sleep 10; tmux kill-session -t \"\$sid\") >/dev/null 2>&1 & disown; echo \"All tests passed. This tmux session will close in 10s...\"; fi; echo; echo \"pytest exited with code: \$status\"; echo \"(You are now in a shell. Press Ctrl-D to close this window.)\"; exec bash -l'" "$target"
}

# Ensure we don't collide with existing sessions
unique_session_name() {
  local base="$1" name="$1" n=1
  while tmux has-session -t "$name" 2>/dev/null; do
    ((n++)); name="${base}-${n}"
  done
  printf "%s" "$name"
}

# Turn a file path into a session base name
#   ./animals/dogs/test_bark.py  -> animals-dogs-test_bark
session_basename_for() {
  local p="$1"
  # normalize to a relative-looking path for naming
  [[ "$p" = /* ]] || p="./${p#./}"
  p="${p%.py}"
  p="${p#./}"
  p="${p//\//-}"
  # Shorter name: drop the "pytest-" prefix
  printf "%s" "$p"
}

# Collect args: files and/or directories to search
declare -a roots=()
declare -a direct_files=()

if (( $# == 0 )); then
  roots=( "." )
else
  for arg in "$@"; do
    if [[ -f "$arg" ]]; then
      # only include python files directly, ignoring any conftest.py
      if [[ "$arg" == *.py && "${arg##*/}" != "conftest.py" ]]; then
        direct_files+=( "$arg" )
      fi
    elif [[ -d "$arg" ]]; then
      roots+=( "$arg" )
    else
      echo "Warning: Skipping non-existent path: $arg" >&2
    fi
  done
  if (( ${#roots[@]} == 0 && ${#direct_files[@]} == 0 )); then
    echo "No valid directories or .py files provided." >&2
    exit 1
  fi
fi

# Build a safe find pipeline:
# find <roots> \( -type d \( -name EX1 -o EX2 ... \) -prune \) -o \( -type f -name "*.py" -print0 \)
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
  cmd+=( ")" -prune ")" -o "(" -type f -name "*.py" ! -name "conftest.py" -print0 ")" )

  printf '%q ' "${cmd[@]}"
}

# Gather recursive .py files from roots (NUL-delimited, sorted)
declare -a found_files=()
if (( ${#roots[@]} )); then
  found_files=()
  while IFS= read -r -d '' f; do
    found_files+=( "$f" )
  done < <(eval "$(build_find_cmd)")
fi

# Combine direct .py files (from args) and found files; sort deterministically
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
if (( ${#direct_files[@]} )); then
  printf '%s\0' "${direct_files[@]}" >> "$tmp"
fi
if (( ${#found_files[@]} )); then
  printf '%s\0' "${found_files[@]}" >> "$tmp"
fi

files=()
while IFS= read -r -d '' f; do
  files+=( "$f" )
done < <(tr '\0' '\n' < "$tmp" | LC_ALL=C sort | tr '\n' '\0')

if (( ${#files[@]} == 0 )); then
  echo "No *.py files found."
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
echo "Usage examples:"
echo "  • Run everything under current dir:     ./\\.parallel_run.sh"
echo "  • Only a folder:                         ./\\.parallel_run.sh test_cats"
echo "  • Multiple roots:                        ./\\.parallel_run.sh tests/unit tests/integration"
echo "  • Specific files:                        ./\\.parallel_run.sh tests/foo.py tests/bar.py"
echo
echo "Tips:"
echo "  • List sessions: tmux ls"
echo "  • Attach:       tmux attach -t <session>"
echo "  • Inside tmux:  tmux switch-client -t <session>"
