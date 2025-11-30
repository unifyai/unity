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

# Wait for completion flag
WAIT_FOR_COMPLETION=0

# Optional filename match (glob-like, e.g., "*_tool_docstring*")
NAME_PATTERN=""

# Project mode: default is shared project (RANDOM_PROJECTS=0)
# With --random-projects: each tmux session gets its own isolated project
RANDOM_PROJECTS=0

# Test category filters (symbolic ↔ eval spectrum)
# With --eval-only: run only tests marked with pytest.mark.eval
# With --symbolic-only: run only tests NOT marked with pytest.mark.eval
EVAL_ONLY=0
SYMBOLIC_ONLY=0

# Cache control
# With --no-cache: disable LLM response caching (UNIFY_CACHE=false)
NO_CACHE=0

# Resolve repo root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

# Parse flags; collect positional args
declare -a POSITIONAL_ARGS=()
while (( "$#" )); do
  case "$1" in
    -w|--wait)
      WAIT_FOR_COMPLETION=1
      shift
      ;;
    -t|--per-test)
      PER_TEST=1
      shift
      ;;
    -m|--match)
      if [[ -n "${2-}" ]]; then
        NAME_PATTERN="$2"
        shift 2
      else
        echo "Error: -m|--match requires a pattern argument (e.g., \"*_tool_docstring*\")." >&2
        exit 2
      fi
      ;;
    --random-projects)
      RANDOM_PROJECTS=1
      shift
      ;;
    --eval-only)
      EVAL_ONLY=1
      shift
      ;;
    --symbolic-only)
      SYMBOLIC_ONLY=1
      shift
      ;;
    --no-cache)
      NO_CACHE=1
      shift
      ;;
    *)
      POSITIONAL_ARGS+=( "$1" )
      shift
      ;;
  esac
done

# Validate mutually exclusive flags
if (( EVAL_ONLY && SYMBOLIC_ONLY )); then
  echo "Error: --eval-only and --symbolic-only are mutually exclusive." >&2
  exit 2
fi

# Build pytest marker filter based on flags
MARKER_FILTER=""
if (( EVAL_ONLY )); then
  MARKER_FILTER="-m eval"
elif (( SYMBOLIC_ONLY )); then
  MARKER_FILTER="-m 'not eval'"
fi
# Reset positional parameters safely under nounset (only expand if set)
set -- ${POSITIONAL_ARGS[@]+"${POSITIONAL_ARGS[@]}"}

# Always operate from the repo root for discovery, regardless of where the script was invoked
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Prepare the shared project (unless using random projects mode)
# ---------------------------------------------------------------------------
if (( ! RANDOM_PROJECTS )); then
  echo "Preparing shared UnityTests project..."
  # Activate virtualenv if available, then run the prepare script
  if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
  fi
  if [[ -f "$SCRIPT_DIR/_prepare_shared_project.py" ]]; then
    python "$SCRIPT_DIR/_prepare_shared_project.py"
  else
    echo "Warning: _prepare_shared_project.py not found." >&2
    echo "Falling back to random projects mode." >&2
    RANDOM_PROJECTS=1
  fi
fi

# Build the command to run in each tmux session
run_cmd() {
  local target="$1"   # pytest target (file path or node id)
  local log_file="$2" # pytest log file path
  local marker_arg="$3"  # optional marker filter (e.g., "-m eval")
  # Build the inner script first with safe %q for path/target, then quote the whole script with %q
  local inner
  local env_exports
  if (( RANDOM_PROJECTS )); then
    # Legacy mode: each session gets its own random project
    env_exports='export UNIFY_TESTS_RAND_PROJ=True UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True'
  else
    # Shared project mode: skip session setup (already done by prepare script)
    env_exports='export UNIFY_SKIP_SESSION_SETUP=True'
  fi
  # Add cache control if --no-cache was specified
  if (( NO_CACHE )); then
    env_exports="$env_exports UNIFY_CACHE=false"
  fi
  # Build pytest command with optional marker filter
  local pytest_cmd
  if [[ -n "$marker_arg" ]]; then
    pytest_cmd=$(printf 'pytest %s %q' "$marker_arg" "$target")
  else
    pytest_cmd=$(printf 'pytest %q' "$target")
  fi
  inner=$(printf '%s PYTEST_LOG_PATH=%q; source ~/unity/.venv/bin/activate && cd %q && %s; status=$?; sname=$(tmux display-message -p -t "$TMUX_PANE" "#{session_name}"); base="$sname"; case "$sname" in "o ✅ "*) base="${sname#o ✅ }" ;; "x ❌ "*) base="${sname#x ❌ }" ;; "? ⏳ "*) base="${sname#? ⏳ }" ;; "✅ "*) base="${sname#✅ }" ;; "❌ "*) base="${sname#❌ }" ;; "⏳ "*) base="${sname#⏳ }" ;; esac; if [ $status -eq 0 ]; then pfx="o ✅"; else pfx="x ❌"; fi; tmux rename-session -t "$sname" "$pfx $base"; if [ $status -eq 0 ]; then sid=$(tmux display-message -p -t "$TMUX_PANE" "#{session_id}"); (sleep 10; tmux kill-session -t "$sid") >/dev/null 2>&1 & disown; echo "All tests passed. This tmux session will close in 10s..."; fi; echo; echo "pytest exited with code: $status"; echo "(You are now in a shell. Press Ctrl-D to close this window.)"; exec bash -l' "$env_exports" "$log_file" "$REPO_ROOT" "$pytest_cmd")
  printf 'bash -lc %q' "$inner"
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
    if [[ "$arg" == *"::"* ]]; then
      # pytest node id: extract base file and suffix; resolve base relative to caller/tests/root
      base="${arg%%::*}"
      suffix="${arg#${base}::}"
      base_path=""
      if [[ -f "$base" ]]; then
        base_path="$base"
      elif [[ -f "$SCRIPT_DIR/$base" ]]; then
        base_path="$SCRIPT_DIR/$base"
      elif [[ -f "$REPO_ROOT/$base" ]]; then
        base_path="$REPO_ROOT/$base"
      fi
      if [[ -n "$base_path" ]]; then
        repo_rel="${base_path#$REPO_ROOT/}"
        if [[ "${repo_rel##*/}" == test_*.py ]]; then
          direct_nodes+=( "${repo_rel}::${suffix}" )
        else
          echo "Warning: Skipping node not under a test_*.py file: $arg" >&2
        fi
      else
        echo "Warning: Skipping non-existent test node (file missing): $arg" >&2
      fi
    elif [[ -f "$arg" || -f "$SCRIPT_DIR/$arg" || -f "$REPO_ROOT/$arg" ]]; then
      # only include Python test files directly (names starting with test_)
      file_path="$arg"
      if [[ ! -f "$file_path" ]]; then
        if [[ -f "$SCRIPT_DIR/$arg" ]]; then
          file_path="$SCRIPT_DIR/$arg"
        else
          file_path="$REPO_ROOT/$arg"
        fi
      fi
      repo_rel="${file_path#$REPO_ROOT/}"
      if [[ "${repo_rel##*/}" == test_*.py ]]; then
        direct_files+=( "$repo_rel" )
      fi
    elif [[ -d "$arg" || -d "$SCRIPT_DIR/$arg" || -d "$REPO_ROOT/$arg" ]]; then
      dir_path="$arg"
      if [[ ! -d "$dir_path" ]]; then
        if [[ -d "$SCRIPT_DIR/$arg" ]]; then
          dir_path="$SCRIPT_DIR/$arg"
        else
          dir_path="$REPO_ROOT/$arg"
        fi
      fi
      repo_rel="${dir_path#$REPO_ROOT/}"
      roots+=( "$repo_rel" )
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
  local marker_arg="$2"  # optional marker filter
  local cmd
  local env_exports
  if (( RANDOM_PROJECTS )); then
    env_exports='export UNIFY_TESTS_RAND_PROJ=True UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True'
  else
    env_exports='export UNIFY_SKIP_SESSION_SETUP=True'
  fi
  # Add cache control if --no-cache was specified
  if (( NO_CACHE )); then
    env_exports="$env_exports UNIFY_CACHE=false"
  fi
  # Build collection command with optional marker filter
  if [[ -n "$marker_arg" ]]; then
    cmd=$(printf '%s; source ~/unity/.venv/bin/activate && pytest --collect-only -q %s %q' "$env_exports" "$marker_arg" "$target")
  else
    cmd=$(printf '%s; source ~/unity/.venv/bin/activate && pytest --collect-only -q %q' "$env_exports" "$target")
  fi
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

# Apply filename pattern filter (matches on basename) if provided
if [[ -n "$NAME_PATTERN" ]]; then
  if (( ${#direct_files[@]} )); then
    tmp_direct=()
    for f in "${direct_files[@]}"; do
      b="${f##*/}"
      if [[ "$b" == $NAME_PATTERN ]]; then
        tmp_direct+=( "$f" )
      fi
    done
    direct_files=( "${tmp_direct[@]}" )
  fi
  if (( ${#found_files[@]} )); then
    tmp_found=()
    for f in "${found_files[@]}"; do
      b="${f##*/}"
      if [[ "$b" == $NAME_PATTERN ]]; then
        tmp_found+=( "$f" )
      fi
    done
    found_files=( "${tmp_found[@]}" )
  fi
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
      done < <(collect_nodes_for_target "$f" "$MARKER_FILTER")
    done
  fi
  if (( ${#found_files[@]} )); then
    for f in "${found_files[@]}"; do
      while IFS= read -r nid; do
        [[ -n "$nid" ]] && printf '%s\0' "$nid" >> "$tmp"
      done < <(collect_nodes_for_target "$f" "$MARKER_FILTER")
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
declare -a session_ids=()
for target in "${files[@]}"; do
  base_sess="$(session_basename_for "$target")"
  session="$(unique_session_name "$base_sess")"

  # Window name = basename without .py
  fname="${target##*/}"
  wname="${fname%.py}"

  # Define log file in .pytest_logs
  mkdir -p "$REPO_ROOT/.pytest_logs"
  log_file=".pytest_logs/${session}.txt"

  # Create the session first (no command), set remain-on-exit, then send the command.
  cmd="$(run_cmd "$target" "$log_file" "$MARKER_FILTER")"

  # Capture session ID to track this specific run robustly
  sid=$(tmux new-session -d -P -F "#{session_id}" -s "$session" -n "$wname" "$cmd")

  pending_name="$(unique_session_name "? ⏳ $session")"
  tmux rename-session -t "$sid" "$pending_name"
  session="$pending_name"

  made_sessions+=( "$session" )
  session_ids+=( "$sid" )
done

echo "Created ${#made_sessions[@]} tmux sessions:"
for s in "${made_sessions[@]}"; do
  echo "  - $s"
done

echo
echo "Trigger:"
echo "  • Run everything under current dir:     ./.parallel_run.sh"
echo "  • Only a folder:                         ./.parallel_run.sh test_cats"
echo "  • Multiple roots:                        ./.parallel_run.sh tests/unit tests/integration"
echo "  • Specific files:                        ./.parallel_run.sh tests/test_foo.py tests/test_bar.py"
echo "  • Specific tests:                        ./.parallel_run.sh tests/test_foo.py::TestA::test_x tests/test_bar.py::test_y"
echo "  • Per-test (dirs/files):                 ./.parallel_run.sh -t tests tests/test_foo.py"
echo "  • Per-test (everything here):            ./.parallel_run.sh -t"
echo "  • Use isolated random projects:          ./.parallel_run.sh --random-projects tests"
echo "  • Run only eval tests:                   ./.parallel_run.sh --eval-only tests"
echo "  • Run only symbolic tests:               ./.parallel_run.sh --symbolic-only tests"
echo "  • Disable LLM caching:                   ./.parallel_run.sh --no-cache tests"
echo
echo "Observe:"
echo "  • Watch sessions: watch -n 0.5 'tmux ls'"
echo "  • List sessions: tmux ls"
echo "  • Attach:       tmux attach -t <session>"
echo "  • Inside tmux:  tmux switch-client -t <session>"

if (( WAIT_FOR_COMPLETION )); then
  echo "Waiting for tests to complete..."

  while true; do
    pending_count=0
    for sid in "${session_ids[@]}"; do
      # Check name of our specific session IDs only
      current_name=$(tmux display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
      # Look for ASCII marker "?" (with or without emoji following) to detect pending state
      if [[ "$current_name" == "?"* ]]; then
        ((pending_count++))
      fi
    done

    if (( pending_count == 0 )); then
      break
    fi
    sleep 1
  done

  echo "All tests completed."

  failures=0
  for sid in "${session_ids[@]}"; do
    current_name=$(tmux display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
    # Look for ASCII marker "x" (with or without emoji following) to detect failure
    if [[ "$current_name" == "x"* ]]; then
      echo "Failure detected in session: $current_name"
      failures=1
    fi
  done

  if (( failures )); then
    echo "Failures detected. Logs are available in .pytest_logs/"
    exit 1
  else
    echo "All tests passed!"
    exit 0
  fi
fi
