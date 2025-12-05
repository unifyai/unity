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

# ---- Terminal-based isolation ----
# Each terminal session (including Cursor agent terminals) gets its own
# isolated tmux server via a unique socket. This prevents agents from
# interfering with each other (e.g., `tmux kill-server` only affects
# the calling terminal's sessions).
#
# The socket is derived from the terminal's TTY device, which is unique
# and stable for each terminal session.
_derive_socket_name() {
  local tty_id
  # Try to get TTY; if not available (non-interactive shell), use a fallback
  tty_id=$(tty 2>/dev/null)
  if [[ "$tty_id" == "not a tty" || -z "$tty_id" || ! "$tty_id" =~ ^/ ]]; then
    # Non-interactive shell: use parent PID chain as fallback
    # This provides some isolation but may not be as stable
    tty_id="pid$$"
  else
    # Interactive shell: use TTY path with slashes replaced
    tty_id=$(echo "$tty_id" | sed 's|/|_|g')
  fi
  echo "unity${tty_id}"
}

TMUX_SOCKET="${UNITY_TEST_SOCKET:-$(_derive_socket_name)}"

# Wrapper for all tmux commands to use our isolated socket
# LC_ALL=en_US.UTF-8 ensures Unicode emojis work in session names
tmux_cmd() {
  LC_ALL=en_US.UTF-8 tmux -L "$TMUX_SOCKET" "$@"
}

# ---- Configurable directory excludes (by name) ----
# Note: 'fixtures' is excluded because those are test data files, not tests themselves.
# They get run explicitly by the test harness (e.g., test_parallel_run tests).
EXCLUDE_DIRS=( .git .hg .svn .venv venv .mypy_cache .pytest_cache __pycache__ .idea .vscode fixtures )

# ---- Modes ----
# Default: one session per file.
# With -t/--per-test: one session per collected pytest node id across provided dirs/files.
PER_TEST=0

# Wait for completion flag and optional timeout
WAIT_FOR_COMPLETION=0
WAIT_TIMEOUT=0  # 0 means no timeout (wait indefinitely)

# Optional filename match (glob-like, e.g., "*_tool_docstring*")
NAME_PATTERN=""

# Test category filters (symbolic ↔ eval spectrum)
# With --eval-only: run only tests marked with pytest.mark.eval
# With --symbolic-only: run only tests NOT marked with pytest.mark.eval
EVAL_ONLY=0
SYMBOLIC_ONLY=0

# Repeat count for statistical sampling
# With --repeat N: run each test N times (useful for eval tests)
REPEAT_COUNT=1

# Environment variable overrides (accumulated via --env KEY=VALUE)
declare -a ENV_OVERRIDES=()

# Tags (accumulated via --tags, shorthand for UNIFY_TEST_TAGS)
declare -a TAGS=()

# Resolve repo root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

# Parse flags; collect positional args
declare -a POSITIONAL_ARGS=()
while (( "$#" )); do
  case "$1" in
    -w|--wait)
      WAIT_FOR_COMPLETION=1
      # Check if next arg is an optional timeout (positive integer)
      if [[ -n "${2-}" && "$2" =~ ^[0-9]+$ && "$2" -ge 1 ]]; then
        WAIT_TIMEOUT="$2"
        shift 2
      else
        shift
      fi
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
    -e|--env)
      if [[ -n "${2-}" && "$2" == *=* ]]; then
        ENV_OVERRIDES+=( "$2" )
        shift 2
      else
        echo "Error: -e|--env requires KEY=VALUE argument (e.g., --env UNIFY_CACHE=false)." >&2
        exit 2
      fi
      ;;
    --eval-only)
      EVAL_ONLY=1
      shift
      ;;
    --symbolic-only)
      SYMBOLIC_ONLY=1
      shift
      ;;
    --repeat)
      if [[ -n "${2-}" && "$2" =~ ^[0-9]+$ && "$2" -ge 1 ]]; then
        REPEAT_COUNT="$2"
        shift 2
      else
        echo "Error: --repeat requires a positive integer argument (e.g., --repeat 5)." >&2
        exit 2
      fi
      ;;
    --tags)
      if [[ -n "${2-}" ]]; then
        # Split on comma and add each tag to TAGS array
        IFS=',' read -ra tag_parts <<< "$2"
        for tag in "${tag_parts[@]}"; do
          [[ -n "$tag" ]] && TAGS+=( "$tag" )
        done
        shift 2
      else
        echo "Error: --tags requires a value (e.g., --tags experiment-1 or --tags \"foo,bar\")." >&2
        exit 2
      fi
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

# ---------------------------------------------------------------------------
# Helper: check if a boolean env var is truthy (via --env flags OR system env)
# Usage: is_env_truthy VAR_NAME
# ---------------------------------------------------------------------------
is_env_truthy() {
  local var_name="$1"
  # Check --env flags first
  for kv in "${ENV_OVERRIDES[@]+"${ENV_OVERRIDES[@]}"}"; do
    case "$kv" in
      "${var_name}=true"|"${var_name}=True"|"${var_name}=1")
        return 0 ;;
      "${var_name}=false"|"${var_name}=False"|"${var_name}=0"|"${var_name}=")
        return 1 ;;
    esac
  done
  # Fall back to system environment variable
  local val="${!var_name:-}"
  case "$val" in
    true|True|1) return 0 ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Helper: get env var value (--env flags take precedence over system env)
# Usage: get_env_value VAR_NAME [DEFAULT]
# ---------------------------------------------------------------------------
get_env_value() {
  local var_name="$1"
  local default="${2:-}"
  # Check --env flags first
  for kv in "${ENV_OVERRIDES[@]+"${ENV_OVERRIDES[@]}"}"; do
    if [[ "$kv" == "${var_name}="* ]]; then
      echo "${kv#${var_name}=}"
      return 0
    fi
  done
  # Fall back to system environment variable
  local val="${!var_name:-$default}"
  echo "$val"
}

# ---------------------------------------------------------------------------
# Helper: check if random projects mode is enabled
# ---------------------------------------------------------------------------
is_random_projects_mode() {
  is_env_truthy "UNIFY_TESTS_RAND_PROJ"
}

# ---------------------------------------------------------------------------
# Helper: build environment exports string from --env overrides, system env, and --tags
# ---------------------------------------------------------------------------
build_env_exports() {
  local exports=""
  # Track which vars are already set via --env flags
  declare -A env_set
  for kv in "${ENV_OVERRIDES[@]+"${ENV_OVERRIDES[@]}"}"; do
    exports="$exports $kv"
    local var_name="${kv%%=*}"
    env_set["$var_name"]=1
  done

  # Propagate relevant system environment variables if not already set via --env
  local propagate_vars=(
    "UNIFY_TESTS_RAND_PROJ"
    "UNIFY_TESTS_DELETE_PROJ_ON_EXIT"
    "UNIFY_SKIP_SESSION_SETUP"
    "UNIFY_CACHE"
    "UNIFY_KEY"
    "UNIFY_BASE_URL"
  )
  for var_name in "${propagate_vars[@]}"; do
    if [[ -z "${env_set[$var_name]:-}" && -n "${!var_name:-}" ]]; then
      exports="$exports ${var_name}=${!var_name}"
    fi
  done

  # Append UNIFY_TEST_TAGS if any tags were specified via --tags
  if (( ${#TAGS[@]} > 0 )); then
    local joined_tags
    joined_tags=$(IFS=','; echo "${TAGS[*]}")
    exports="$exports UNIFY_TEST_TAGS=$joined_tags"
  elif [[ -z "${env_set[UNIFY_TEST_TAGS]:-}" && -n "${UNIFY_TEST_TAGS:-}" ]]; then
    # Propagate from system env if not set via --tags or --env
    exports="$exports UNIFY_TEST_TAGS=$UNIFY_TEST_TAGS"
  fi
  echo "$exports"
}

# Reset positional parameters safely under nounset (only expand if set)
set -- ${POSITIONAL_ARGS[@]+"${POSITIONAL_ARGS[@]}"}

# Always operate from the repo root for discovery, regardless of where the script was invoked
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Prepare the shared project (unless using random projects mode)
# ---------------------------------------------------------------------------
if is_random_projects_mode; then
  echo "Random projects mode detected; skipping shared project preparation..."
else
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
    ENV_OVERRIDES+=( "UNIFY_TESTS_RAND_PROJ=True" "UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True" )
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
  if is_random_projects_mode; then
    # Random projects mode: each session gets its own project
    env_exports='export UNIFY_TESTS_RAND_PROJ=True UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True'
  else
    # Shared project mode: skip session setup (already done by prepare script)
    env_exports='export UNIFY_SKIP_SESSION_SETUP=True'
  fi
  # Append user-provided --env overrides
  local user_overrides
  user_overrides="$(build_env_exports)"
  if [[ -n "$user_overrides" ]]; then
    env_exports="$env_exports$user_overrides"
  fi
  # Build pytest command with optional marker filter
  local pytest_cmd
  if [[ -n "$marker_arg" ]]; then
    pytest_cmd=$(printf 'pytest %s %q' "$marker_arg" "$target")
  else
    pytest_cmd=$(printf 'pytest %q' "$target")
  fi
  # Build inner command with socket name directly interpolated (not via env var)
  # This ensures tmux commands target the correct isolated server
  # Note: LC_ALL=en_US.UTF-8 is required for Unicode emoji support in tmux session names
  inner=$(printf '%s PYTEST_LOG_PATH=%q; source ~/unity/.venv/bin/activate && cd %q && %s; status=$?; sname=$(LC_ALL=en_US.UTF-8 tmux -L %q display-message -p -t "$TMUX_PANE" "#{session_name}"); base="$sname"; case "$sname" in "d ✅ "*) base="${sname#d ✅ }" ;; "f ❌ "*) base="${sname#f ❌ }" ;; "r ⏳ "*) base="${sname#r ⏳ }" ;; esac; if [ $status -eq 0 ]; then pfx="d ✅"; else pfx="f ❌"; fi; LC_ALL=en_US.UTF-8 tmux -L %q rename-session -t "$sname" "$pfx $base"; if [ $status -eq 0 ]; then sid=$(LC_ALL=en_US.UTF-8 tmux -L %q display-message -p -t "$TMUX_PANE" "#{session_id}"); (sleep 10; LC_ALL=en_US.UTF-8 tmux -L %q kill-session -t "$sid") >/dev/null 2>&1 & disown; echo "All tests passed. This tmux session will close in 10s..."; fi; echo; echo "pytest exited with code: $status"; echo "(You are now in a shell. Press Ctrl-D to close this window.)"; exec bash -l' "$env_exports" "$log_file" "$REPO_ROOT" "$pytest_cmd" "$TMUX_SOCKET" "$TMUX_SOCKET" "$TMUX_SOCKET" "$TMUX_SOCKET")
  printf 'bash -lc %q' "$inner"
}

# Ensure we don't collide with existing sessions
unique_session_name() {
  local base="$1" name="$1" n=1
  while tmux_cmd has-session -t "$name" 2>/dev/null; do
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
  if is_random_projects_mode; then
    env_exports='export UNIFY_TESTS_RAND_PROJ=True UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True'
  else
    env_exports='export UNIFY_SKIP_SESSION_SETUP=True'
  fi
  # Append user-provided --env overrides
  local user_overrides
  user_overrides="$(build_env_exports)"
  if [[ -n "$user_overrides" ]]; then
    env_exports="$env_exports$user_overrides"
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

# Expand targets for repeat runs (statistical sampling)
if (( REPEAT_COUNT > 1 )); then
  original_files=( "${files[@]}" )
  files=()
  for (( r=1; r<=REPEAT_COUNT; r++ )); do
    for f in "${original_files[@]}"; do
      files+=( "$f" )
    done
  done
  echo "Repeating each test $REPEAT_COUNT times (${#files[@]} total sessions from ${#original_files[@]} unique targets)"
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
  sid=$(tmux_cmd new-session -d -P -F "#{session_id}" -s "$session" -n "$wname" "$cmd")

  pending_name="$(unique_session_name "r ⏳ $session")"
  tmux_cmd rename-session -t "$sid" "$pending_name"
  session="$pending_name"

  made_sessions+=( "$session" )
  session_ids+=( "$sid" )
done

echo "Created ${#made_sessions[@]} tmux sessions (socket: $TMUX_SOCKET):"
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
echo "  • Set environment variables:             ./.parallel_run.sh --env UNIFY_CACHE=false tests"
echo "  • Multiple env vars:                     ./.parallel_run.sh -e UNIFY_CACHE=false -e UNIFY_DELETE_CONTEXT_ON_EXIT=true tests"
echo "  • Tag test runs:                         ./.parallel_run.sh --tags experiment-1 tests"
echo "  • Multiple tags:                         ./.parallel_run.sh --tags \"model-compare,gpt-4o\" tests"
echo "  • Run only eval tests:                   ./.parallel_run.sh --eval-only tests"
echo "  • Run only symbolic tests:               ./.parallel_run.sh --symbolic-only tests"
echo "  • Repeat tests for sampling:             ./.parallel_run.sh --repeat 5 --eval-only tests"
echo
echo "Observe (this terminal's sessions only):"
echo "  • Watch sessions:  tests/watch_tests.sh"
echo "  • List sessions:   tmux -L $TMUX_SOCKET ls"
echo "  • Attach:          tmux -L $TMUX_SOCKET attach -t <session>"
echo
echo "See all terminals' tests: tests/.list_all_tests.sh"

if (( WAIT_FOR_COMPLETION )); then
  if (( WAIT_TIMEOUT > 0 )); then
    echo "Waiting for tests to complete (timeout: ${WAIT_TIMEOUT}s)..."
  else
    echo "Waiting for tests to complete..."
  fi

  wait_start=$(date +%s)
  timed_out=0
  while true; do
    pending_count=0
    for sid in "${session_ids[@]}"; do
      # Check name of our specific session IDs only
      current_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
      # Look for "?" prefix to detect pending state
      if [[ "$current_name" == "?"* ]]; then
        ((pending_count++))
      fi
    done

    if (( pending_count == 0 )); then
      break
    fi

    # Check timeout if specified
    if (( WAIT_TIMEOUT > 0 )); then
      elapsed=$(( $(date +%s) - wait_start ))
      if (( elapsed >= WAIT_TIMEOUT )); then
        timed_out=1
        echo "Timeout reached after ${WAIT_TIMEOUT}s. ${pending_count} session(s) still running."
        break
      fi
    fi

    sleep 1
  done

  if (( timed_out )); then
    echo "Tests did not complete within timeout. Check tmux sessions manually."
    exit 2
  fi

  echo "All tests completed."

  failures=0
  for sid in "${session_ids[@]}"; do
    current_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
    # Look for "x" prefix to detect failure
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
