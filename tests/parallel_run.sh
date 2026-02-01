#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory first (needed for relative path resolution)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# Optionally source environment from the repo root's .env
# This allows storing secrets/config like UNIFY_KEY in the repo root `.env` (not committed).
_ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$_ENV_FILE" ]; then
  # shellcheck disable=SC1090
  set -a
  . "$_ENV_FILE"
  set +a
fi
unset _ENV_FILE

# Source common utilities (socket derivation, locale, timeout handling)
source "$SCRIPT_DIR/_shell_common.sh"

# Source shared argument parsing (used by both parallel_run.sh and parallel_cloud_run.sh)
source "$SCRIPT_DIR/_parse_args.sh"

# ---- Increase file descriptor limit ----
# Parallel tests open many network connections. Each connection uses a file
# descriptor. macOS defaults to 256 per process, which is easily exceeded.
# This setting is inherited by all child processes (tmux sessions, pytest).
ulimit -n 8192 2>/dev/null || true

TMUX_SOCKET="$UNITY_TMUX_SOCKET"

# ---- Log directory naming ----
# Log subdirectories use a datetime-prefixed format for natural time-based
# ordering in the filesystem. Format: YYYY-MM-DDTHH-MM-SS_{socket_name}
# This makes it easy to find recent test runs while preserving terminal isolation.
_derive_log_subdir() {
  local socket_name="$1"
  local datetime
  datetime=$(date +"%Y-%m-%dT%H-%M-%S")
  echo "${datetime}_${socket_name}"
}

# Generate log subdir once at script start (stable for this run)
LOG_SUBDIR="${UNITY_LOG_SUBDIR:-$(_derive_log_subdir "$TMUX_SOCKET")}"

# Wrapper for all tmux commands to use our isolated socket
# LC_ALL=en_US.UTF-8 ensures Unicode emojis work in session names
tmux_cmd() {
  LC_ALL=en_US.UTF-8 tmux -L "$TMUX_SOCKET" "$@"
}

# ---- Cleanup on interrupt ----
# Track session IDs for cleanup on SIGINT/SIGTERM
declare -a CREATED_SESSION_IDS=()

# ---- Inline pass/fail reporting ----
# Track which sessions we've already reported completion for (newline-separated list)
REPORTED_COMPLETIONS=""

# Check if a session ID has been reported
_is_reported() {
  local sid="$1"
  [[ "$REPORTED_COMPLETIONS" == *"${sid}"* ]]
}

# Mark a session ID as reported
_mark_reported() {
  local sid="$1"
  REPORTED_COMPLETIONS="${REPORTED_COMPLETIONS}${sid}:"
}

# Report any sessions that have completed since last check
# Prints pass/fail status inline during the drip-feed phase
# Also records duration and cache stats for end-of-run sorted output
report_completed_sessions() {
  # Guard against empty array (set -u treats empty array expansion as unbound)
  (( ${#CREATED_SESSION_IDS[@]} == 0 )) && return 0

  for sid in "${CREATED_SESSION_IDS[@]}"; do
    # Skip if already reported
    _is_reported "$sid" && continue

    # Get current session name (may fail if session was killed)
    local current_name
    current_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
    [[ -z "$current_name" ]] && continue

    # Check for completion (passed or failed prefix)
    case "$current_name" in
      "p ✅ "*)
        local base="${current_name#p ✅ }"
        echo "  - p ✅ $base"
        _mark_reported "$sid"
        # Record duration and cache stats for sorted output
        if [[ -n "${START_TIMES_FILE:-}" && -f "$START_TIMES_FILE" ]]; then
          local start_time end_time duration hits misses
          start_time=$(grep "^$sid " "$START_TIMES_FILE" 2>/dev/null | cut -d' ' -f2)
          if [[ -n "$start_time" ]]; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            # Read cache stats from temp file (written by inner script)
            hits=0
            misses=0
            local stats_file="/tmp/parallel_run_cache_${sid}.txt"
            if [[ -f "$stats_file" ]]; then
              IFS='|' read -r hits misses < "$stats_file" 2>/dev/null || true
              rm -f "$stats_file"
            fi
            echo "$duration|pass|$hits|$misses|$base" >> "$RESULTS_FILE"
          fi
        fi
        ;;
      "f ❌ "*)
        local base="${current_name#f ❌ }"
        echo "  - f ❌ $base"
        _mark_reported "$sid"
        # Record duration and cache stats for sorted output
        if [[ -n "${START_TIMES_FILE:-}" && -f "$START_TIMES_FILE" ]]; then
          local start_time end_time duration hits misses
          start_time=$(grep "^$sid " "$START_TIMES_FILE" 2>/dev/null | cut -d' ' -f2)
          if [[ -n "$start_time" ]]; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            # Read cache stats from temp file (written by inner script)
            hits=0
            misses=0
            local stats_file="/tmp/parallel_run_cache_${sid}.txt"
            if [[ -f "$stats_file" ]]; then
              IFS='|' read -r hits misses < "$stats_file" 2>/dev/null || true
              rm -f "$stats_file"
            fi
            echo "$duration|fail|$hits|$misses|$base" >> "$RESULTS_FILE"
          fi
        fi
        ;;
    esac
  done
}

_cleanup_sessions() {
  local sig="${1:-}"
  if (( ${#CREATED_SESSION_IDS[@]} > 0 )); then
    echo ""
    echo "Caught signal${sig:+ ($sig)}. Cleaning up ${#CREATED_SESSION_IDS[@]} session(s)..."
    for sid in "${CREATED_SESSION_IDS[@]}"; do
      # Send SIGTERM to allow graceful Python shutdown, then kill session
      # First, send SIGTERM to all processes in the session
      tmux_cmd list-panes -t "$sid" -F '#{pane_pid}' 2>/dev/null | while read -r pid; do
        if [[ -n "$pid" ]]; then
          # Kill the entire process group to catch child processes
          kill -TERM "-$pid" 2>/dev/null || true
        fi
      done
      # Brief wait for graceful shutdown
      sleep 0.1
      # Then kill the tmux session
      tmux_cmd kill-session -t "$sid" 2>/dev/null || true
    done
    echo "Cleanup complete."
  fi
}

# Set up signal handlers for graceful cleanup
trap '_cleanup_sessions INT; exit 130' INT
trap '_cleanup_sessions TERM; exit 143' TERM

# ---- Configurable directory excludes (by name) ----
# Note: 'fixtures' is excluded because those are test data files, not tests themselves.
# They get run explicitly by the test harness (e.g., test_parallel_run tests).
EXCLUDE_DIRS=( .git .hg .svn .venv venv .mypy_cache .pytest_cache __pycache__ .idea .vscode fixtures )

# Resolve repo root (parent of this script's directory)
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

# Parse arguments using shared helper
# Returns: 0=success, 1=help requested, 2=error
parse_test_args "$@"
_parse_result=$?
if (( _parse_result == 1 )); then
  # Help requested
  HELP_SCRIPT_NAME="parallel_run.sh"
  print_help
  exit 0
elif (( _parse_result == 2 )); then
  # Error (already printed)
  exit 2
fi
unset _parse_result

# For backward compatibility, expose _NUM_CORES (used in some places)
_NUM_CORES=$DETECTED_CPU_CORES

# ---------------------------------------------------------------------------
# Local Orchestra Setup
# ---------------------------------------------------------------------------
# UNIFY_BASE_URL is the single source of truth:
# - Unset or localhost (127.0.0.1/localhost): use local orchestra
# - Any other URL: use it directly (staging, production, etc.)
#
# Local orchestra is started via the orchestra repo's scripts/local.sh.
# Set ORCHESTRA_REPO_PATH to override the default location (../orchestra).
#
# WORKTREE HANDLING:
# When running from a git worktree (e.g., Cursor Background Agents), we don't
# restart orchestra just to change log directories. Instead, we create symlinks
# from the worktree's logs/ to wherever orchestra is currently logging. This
# avoids disrupting concurrent tests in the main repo or other worktrees.

_is_local_url() {
  local url="${1:-}"
  [[ -z "$url" ]] && return 0  # Unset = local
  [[ "$url" == *"127.0.0.1"* || "$url" == *"localhost"* ]]
}

_is_git_worktree() {
  # In git worktrees, .git is a file (containing "gitdir: /path/..."), not a directory
  [[ -f "$REPO_ROOT/.git" ]]
}

_create_orchestra_log_symlinks() {
  # Create symlinks from worktree's log directories to the main repo's log directories.
  # This ensures all orchestra logs (and OTEL traces) go to a single shared location,
  # while allowing the worktree to "see" them via symlinks.
  #
  # Only called when:
  # 1. We're in a git worktree
  # 2. Orchestra is already running with logs pointing elsewhere

  local config_file="/tmp/orchestra-local-server.config"

  [[ -f "$config_file" ]] || return 0

  local current_log_dir current_otel_dir
  current_log_dir=$(grep "^ORCHESTRA_LOG_DIR=" "$config_file" 2>/dev/null | cut -d= -f2-)
  current_otel_dir=$(grep "^ORCHESTRA_OTEL_LOG_DIR=" "$config_file" 2>/dev/null | cut -d= -f2-)

  # Ensure logs/ directory exists in worktree
  mkdir -p "$REPO_ROOT/logs"

  # Symlink logs/orchestra/ → main repo's logs/orchestra/
  # ORCHESTRA_LOG_DIR includes timestamp subfolder, so we go up one level
  if [[ -n "$current_log_dir" ]]; then
    local target_dir
    target_dir=$(dirname "$current_log_dir")  # Strip timestamp subfolder
    local link_path="$REPO_ROOT/logs/orchestra"

    if [[ -L "$link_path" ]]; then
      # Symlink exists - update if pointing elsewhere
      local current_target
      current_target=$(readlink "$link_path")
      if [[ "$current_target" != "$target_dir" ]]; then
        rm "$link_path"
        ln -s "$target_dir" "$link_path"
        echo "  Updated symlink: logs/orchestra → $target_dir"
      fi
    elif [[ ! -e "$link_path" ]]; then
      # Create new symlink
      ln -s "$target_dir" "$link_path"
      echo "  Created symlink: logs/orchestra → $target_dir"
    fi
    # If it's a real directory, leave it alone (user may have customized)
  fi

  # Symlink logs/all/ → main repo's logs/all/ (for OTEL trace correlation)
  # This ensures unity/unify/unillm spans from worktree end up in same dir as orchestra spans
  if [[ -n "$current_otel_dir" ]]; then
    local link_path="$REPO_ROOT/logs/all"

    if [[ -L "$link_path" ]]; then
      local current_target
      current_target=$(readlink "$link_path")
      if [[ "$current_target" != "$current_otel_dir" ]]; then
        rm "$link_path"
        ln -s "$current_otel_dir" "$link_path"
        echo "  Updated symlink: logs/all → $current_otel_dir"
      fi
    elif [[ ! -e "$link_path" ]]; then
      ln -s "$current_otel_dir" "$link_path"
      echo "  Created symlink: logs/all → $current_otel_dir"
    fi
  fi
}

# Resolve orchestra repo path (default: sibling directory)
# For worktrees, look relative to the MAIN repo, not the worktree
_orchestra_search_base="$REPO_ROOT"
if _git_common_dir="$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null)"; then
  # git rev-parse --git-common-dir returns a relative path ".git" for regular repos,
  # but an absolute path for worktrees. Normalize to absolute for consistent comparison.
  if [[ "$_git_common_dir" != /* ]]; then
    _git_common_dir="$REPO_ROOT/$_git_common_dir"
  fi
  # If git-common-dir differs from $REPO_ROOT/.git, we're in a worktree
  # Use the main repo's parent as the search base for orchestra
  _main_repo_git="${_git_common_dir%/}"
  if [[ "$_main_repo_git" != "$REPO_ROOT/.git" ]]; then
    _orchestra_search_base="$(dirname "$_main_repo_git")"
  fi
fi
_orchestra_repo_path="${ORCHESTRA_REPO_PATH:-$_orchestra_search_base/../orchestra}"
_local_orchestra_script="$_orchestra_repo_path/scripts/local.sh"
unset _git_common_dir _main_repo_git _orchestra_search_base

if _is_local_url "${UNIFY_BASE_URL:-}"; then
  if [[ -x "$_local_orchestra_script" ]]; then
    # Set up orchestra configuration for tests
    export ORCHESTRA_SEED_USER=1
    export ORCHESTRA_TEST_USER_ID="${ORCHESTRA_TEST_USER_ID:-unity-test-user-001}"
    export ORCHESTRA_TEST_EMAIL="${ORCHESTRA_TEST_EMAIL:-unity-test@debug.local}"

    # Set up log directories (created lazily by local.sh)
    _orchestra_logs_dir="$REPO_ROOT/logs/orchestra"
    _timestamp="$(date +%Y-%m-%dT%H-%M-%S)"
    export ORCHESTRA_LOG_DIR="$_orchestra_logs_dir/$_timestamp"
    export ORCHESTRA_OTEL_LOG_DIR="$REPO_ROOT/logs/all"

    # Check if local orchestra is already running
    if _local_url=$("$_local_orchestra_script" check 2>/dev/null); then
      # Orchestra is running - check if logging config matches what we need
      _config_file="/tmp/orchestra-local-server.config"
      _needs_restart=false

      if [[ -f "$_config_file" ]]; then
        # Read current config and compare with desired logging dirs
        _current_log_dir=$(grep "^ORCHESTRA_LOG_DIR=" "$_config_file" 2>/dev/null | cut -d= -f2-)
        _current_otel_dir=$(grep "^ORCHESTRA_OTEL_LOG_DIR=" "$_config_file" 2>/dev/null | cut -d= -f2-)

        # Check if OTEL dir points to our logs/all directory
        if [[ "$_current_otel_dir" != "$ORCHESTRA_OTEL_LOG_DIR" ]]; then
          _needs_restart=true
        fi
        # Check if per-request log dir points inside our logs/orchestra directory
        if [[ -z "$_current_log_dir" || "$_current_log_dir" != "$_orchestra_logs_dir"/* ]]; then
          _needs_restart=true
        fi
      else
        # No config file means orchestra was started without our logging setup
        _needs_restart=true
      fi

      if [[ "$_needs_restart" == "true" ]]; then
        if _is_git_worktree; then
          # WORKTREE MODE: Don't restart orchestra (would disrupt other worktrees/main repo).
          # Instead, create symlinks so logs appear in expected locations.
          echo "Worktree detected: Using existing orchestra, creating log symlinks..."
          _create_orchestra_log_symlinks
          export UNIFY_BASE_URL="$_local_url"
        else
          # MAIN REPO MODE: Restart to pick up logging config. The restart wipes
          # the database, which is intentional for test runs to ensure isolation.
          _original_url="$_local_url"
          echo "Restarting orchestra to apply logging configuration..."
          "$_local_orchestra_script" restart >/dev/null 2>&1 || true
          if _local_url=$("$_local_orchestra_script" check 2>/dev/null); then
            echo "Using local orchestra: $_local_url"
            export UNIFY_BASE_URL="$_local_url"
          else
            echo "Warning: Orchestra restart failed, using existing instance (logging may not work)" >&2
            export UNIFY_BASE_URL="$_original_url"
          fi
          unset _original_url
        fi
      else
        # Logging already configured correctly, reuse existing instance
        # Update ORCHESTRA_LOG_DIR to match what's currently configured
        export ORCHESTRA_LOG_DIR="$_current_log_dir"
        echo "Local orchestra already running with logging enabled: $_local_url"
        export UNIFY_BASE_URL="$_local_url"
      fi
      unset _config_file _needs_restart _current_log_dir _current_otel_dir
    else
      # Not running - need to start it
      # Stop any stale orchestra state first
      "$_local_orchestra_script" stop >/dev/null 2>&1 || true

      # Remove any existing PostgreSQL container so we get fresh one with correct max_connections
      _db_port="${ORCHESTRA_DB_PORT:-5432}"
      for _container in $(docker ps -a --filter "publish=${_db_port}" --format "{{.Names}}" 2>/dev/null); do
        docker stop "$_container" >/dev/null 2>&1 || true
        docker rm "$_container" >/dev/null 2>&1 || true
      done
      unset _container

      # Wait for DB port to be fully released (Docker Desktop can be slow)
      if lsof -i ":$_db_port" &>/dev/null; then
        echo "Waiting for port $_db_port to be released..."
        _max_wait=30
        _waited=0
        while lsof -i ":$_db_port" &>/dev/null && (( _waited < _max_wait )); do
          sleep 1
          (( ++_waited ))
        done
        echo "Port $_db_port released."
      fi
      unset _db_port _max_wait _waited

      echo "Starting local orchestra..."
      if "$_local_orchestra_script" start >/dev/null 2>&1; then
        if _local_url=$("$_local_orchestra_script" check 2>/dev/null); then
          echo "Using local orchestra: $_local_url"
          export UNIFY_BASE_URL="$_local_url"
        else
          echo "Warning: Local orchestra started but not responding" >&2
        fi
      else
        echo "Warning: Could not start local orchestra" >&2
      fi
    fi
  else
    echo "Warning: Orchestra script not found at $_local_orchestra_script" >&2
    echo "  Set ORCHESTRA_REPO_PATH or clone orchestra repo to ../orchestra" >&2
  fi
  unset _local_url _orchestra_logs_dir _timestamp
else
  echo "Using remote orchestra: $UNIFY_BASE_URL"
fi
unset _orchestra_repo_path _local_orchestra_script

# ---------------------------------------------------------------------------
# Communication Service URL Setup
# ---------------------------------------------------------------------------
# UNITY_COMMS_URL is auto-selected based on git branch if not explicitly set:
# - main branch → production: https://unity-comms-app-262420637606.us-central1.run.app
# - other branches → staging: https://unity-comms-app-staging-262420637606.us-central1.run.app
#
# This mirrors the CI behavior in .github/workflows/tests.yml
if [[ -z "${UNITY_COMMS_URL:-}" ]]; then
  _current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
  if [[ "$_current_branch" == "main" ]]; then
    export UNITY_COMMS_URL="https://unity-comms-app-262420637606.us-central1.run.app"
    echo "Using production communication service (main branch)"
  else
    export UNITY_COMMS_URL="https://unity-comms-app-staging-262420637606.us-central1.run.app"
    echo "Using staging communication service (branch: $_current_branch)"
  fi
  unset _current_branch
else
  echo "Using communication service: $UNITY_COMMS_URL"
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
# Helper: delete the shared project (script-level, not per-session)
# Used for UNIFY_TESTS_DELETE_PROJ_ON_START/EXIT in shared project mode
# ---------------------------------------------------------------------------
delete_shared_project() {
  local phase="$1"  # "start" or "exit"
  echo "Deleting project '${UNIFY_PROJECT:-UnityTests}'..."
  "$VENV_PY" - << 'PYEOF'
import os
import sys
try:
    import unify
    project_name = os.environ.get("UNIFY_PROJECT", "UnityTests")
    try:
        unify.delete_project(project_name, missing_ok=False)
        print(f"Deleted project '{project_name}'")
    except Exception:
        print(f"Project '{project_name}' did not exist, skipping deletion")
except ImportError:
    print("Warning: unify module not available, skipping project deletion")
    sys.exit(0)
PYEOF
}

# ---------------------------------------------------------------------------
# Helper: check if a var name is in the --env overrides
# Usage: is_var_in_env_overrides VAR_NAME
# ---------------------------------------------------------------------------
is_var_in_env_overrides() {
  local var_name="$1"
  for kv in "${ENV_OVERRIDES[@]+"${ENV_OVERRIDES[@]}"}"; do
    if [[ "$kv" == "${var_name}="* ]]; then
      return 0
    fi
  done
  return 1
}

# ---------------------------------------------------------------------------
# Helper: build environment exports string from --env overrides, system env, and --tags
# ---------------------------------------------------------------------------
build_env_exports() {
  local exports=""

  # Always export the socket name for tmux isolation
  exports="$exports UNITY_TEST_SOCKET=$TMUX_SOCKET"

  # Export the log subdir for datetime-prefixed log directory naming
  exports="$exports UNITY_LOG_SUBDIR=$LOG_SUBDIR"

  # ---------------------------------------------------------------------------
  # OpenTelemetry Configuration for Cross-Repo Full-Stack Traces
  # ---------------------------------------------------------------------------
  # Enable OTEL tracing across all four repos (unity, unify, unillm, orchestra)
  # so spans from a single test are aggregated into one {trace_id}.jsonl file.
  # All repos write to logs/all/ (shared directory) - per-test isolation is
  # provided by unique trace_ids, not subdirectories. This allows Orchestra
  # (a persistent server) to participate in cross-process traces.
  local otel_log_dir="$REPO_ROOT/logs/all"

  # Enable OTEL master switches (unless explicitly disabled via --env)
  if ! is_var_in_env_overrides "UNITY_OTEL"; then
    exports="$exports UNITY_OTEL=true"
  fi
  if ! is_var_in_env_overrides "UNIFY_OTEL"; then
    exports="$exports UNIFY_OTEL=true"
  fi
  if ! is_var_in_env_overrides "UNILLM_OTEL"; then
    exports="$exports UNILLM_OTEL=true"
  fi

  # Point all repos to the unified OTEL log directory (unless explicitly set via --env)
  # All repos write {trace_id}.jsonl files; shared directory = unified traces
  if ! is_var_in_env_overrides "UNITY_OTEL_LOG_DIR"; then
    exports="$exports UNITY_OTEL_LOG_DIR=$otel_log_dir"
  fi
  if ! is_var_in_env_overrides "UNIFY_OTEL_LOG_DIR"; then
    exports="$exports UNIFY_OTEL_LOG_DIR=$otel_log_dir"
  fi
  if ! is_var_in_env_overrides "UNILLM_OTEL_LOG_DIR"; then
    exports="$exports UNILLM_OTEL_LOG_DIR=$otel_log_dir"
  fi
  # Note: ORCHESTRA_OTEL_LOG_DIR is set during local orchestra setup above

  # Add all --env flag overrides
  for kv in "${ENV_OVERRIDES[@]+"${ENV_OVERRIDES[@]}"}"; do
    exports="$exports $kv"
  done

  # Propagate relevant system environment variables if not already set via --env
  # Note: UNIFY_TESTS_DELETE_PROJ_ON_START and UNIFY_TESTS_DELETE_PROJ_ON_EXIT are intentionally
  # NOT propagated to individual sessions. They are handled at the script level to avoid race
  # conditions where multiple sessions try to delete the shared project simultaneously.
  # Exception: In random projects mode, deletion is safe per-session (handled in run_cmd).
  local propagate_vars="UNIFY_TESTS_RAND_PROJ UNIFY_SKIP_SESSION_SETUP UNIFY_CACHE UNIFY_KEY UNIFY_BASE_URL UNITY_COMMS_URL UNITY_SKIP_SHARED_PROJECT_PREP PYTHONPATH ANTHROPIC_API_KEY OPENAI_API_KEY"
  for var_name in $propagate_vars; do
    if ! is_var_in_env_overrides "$var_name" && [[ -n "${!var_name:-}" ]]; then
      # Quote values containing special characters (paths, URLs with colons/slashes)
      exports="$exports ${var_name}='${!var_name}'"
    fi
  done

  # Append UNIFY_TEST_TAGS if any tags were specified via --tags
  if (( ${#TAGS[@]} > 0 )); then
    local joined_tags
    joined_tags=$(IFS=','; echo "${TAGS[*]}")
    exports="$exports UNIFY_TEST_TAGS=$joined_tags"
  elif ! is_var_in_env_overrides "UNIFY_TEST_TAGS" && [[ -n "${UNIFY_TEST_TAGS:-}" ]]; then
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
# Worktree dependency symlinks (for local editable packages)
# ---------------------------------------------------------------------------
# pyproject.toml references ../unify and ../unillm as editable local packages.
# In the main repo at ~/unity, these resolve to ~/unify and ~/unillm.
# In a worktree at ~/.cursor/worktrees/unity/xyz, they resolve to
# ~/.cursor/worktrees/unity/unify which doesn't exist by default.
#
# This function creates symlinks at the worktree parent level so that
# `uv sync` works correctly in any worktree without manual setup.
ensure_worktree_dependency_symlinks() {
  local git_file="$REPO_ROOT/.git"

  # Only applies to git worktrees (where .git is a file, not a directory)
  if [[ ! -f "$git_file" ]]; then
    return 0
  fi

  # Parse the gitdir from .git file (format: "gitdir: /path/to/main/.git/worktrees/name")
  local gitdir
  gitdir=$(sed 's/^gitdir: //' "$git_file")

  # Derive the main repo path: /main/repo/.git/worktrees/name -> /main/repo
  local main_repo
  main_repo=$(dirname "$(dirname "$(dirname "$gitdir")")")

  # The worktree parent is where all worktrees live (e.g., ~/.cursor/worktrees/unity/)
  local worktree_parent
  worktree_parent=$(dirname "$REPO_ROOT")

  # For each local dependency, ensure a symlink exists at the worktree parent level
  local deps=("unify" "unillm")
  for dep in "${deps[@]}"; do
    local target="$main_repo/../$dep"  # e.g., ~/unify (sibling of main repo)
    local link="$worktree_parent/$dep"  # e.g., ~/.cursor/worktrees/unity/unify

    # Resolve to absolute path
    if [[ -d "$target" ]]; then
      target=$(cd "$target" && pwd -P)
    else
      # Local dependency doesn't exist; skip (will fail later in uv sync with clear error)
      continue
    fi

    # Create symlink if missing or pointing to wrong location
    if [[ -L "$link" ]]; then
      local current_target
      current_target=$(readlink "$link")
      if [[ "$current_target" != "$target" ]]; then
        echo "Updating worktree symlink: $link -> $target" >&2
        rm "$link"
        ln -s "$target" "$link"
      fi
    elif [[ ! -e "$link" ]]; then
      echo "Creating worktree symlink: $link -> $target" >&2
      ln -s "$target" "$link"
    fi
  done
}

# Ensure symlinks before venv setup (silent if not in a worktree)
ensure_worktree_dependency_symlinks

# ---------------------------------------------------------------------------
# Python environment (uv + repo-local .venv)
# ---------------------------------------------------------------------------
# This script is used in local dev, CI, and Cursor Cloud Agents.
# For portability, avoid hardcoding home-directory venv paths or relying on `python`
# being present on PATH. Instead, bootstrap and use the repo-local `.venv`.
VENV_DIR="$REPO_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"
UV_BIN=""

ensure_uv() {
  # Prefer an existing uv on PATH.
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return 0
  fi
  # Common pip --user install location.
  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
    return 0
  fi

  echo "uv not found; installing via pip (user install)..." >&2
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found; cannot install uv." >&2
    return 1
  fi
  if ! python3 -m pip install --user uv; then
    echo "Error: failed to install uv via pip." >&2
    return 1
  fi

  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return 0
  fi
  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
    return 0
  fi

  echo "Error: uv was installed but is not discoverable (PATH may be missing ~/.local/bin)." >&2
  echo "Try adding ~/.local/bin to PATH, or install uv manually: https://github.com/astral-sh/uv" >&2
  return 1
}

ensure_project_venv() {
  # Fast path: venv exists and can run pytest.
  if [[ -x "$VENV_PY" ]]; then
    if "$VENV_PY" -m pytest --version >/dev/null 2>&1; then
      return 0
    fi
  fi

  ensure_uv || return 1

  echo "Bootstrapping project virtualenv with uv (uv sync --all-groups)..." >&2
  if ! UV_PROJECT_ENVIRONMENT="$VENV_DIR" "$UV_BIN" sync --all-groups; then
    echo "Error: 'uv sync --all-groups' failed." >&2
    return 1
  fi

  if [[ ! -x "$VENV_PY" ]]; then
    echo "Error: expected venv python at '$VENV_PY' after uv sync." >&2
    return 1
  fi
  if ! "$VENV_PY" -m pytest --version >/dev/null 2>&1; then
    echo "Error: pytest is not available in the project venv after uv sync." >&2
    return 1
  fi
}

if ! ensure_project_venv; then
  exit 1
fi

# ---------------------------------------------------------------------------
# Prepare the shared project (unless using random projects mode or skipped)
# ---------------------------------------------------------------------------
# UNITY_SKIP_SHARED_PROJECT_PREP: When set, skip the heavyweight project
# preparation entirely. Useful for:
# - Nested parallel_run.sh calls inside tests (the outer call already prepared)
# - Running fixture tests that don't need the real UnityTests project
if [[ -n "${UNITY_SKIP_SHARED_PROJECT_PREP:-}" ]]; then
  echo "Skipping shared project preparation (UNITY_SKIP_SHARED_PROJECT_PREP set)..."
elif is_random_projects_mode; then
  echo "Random projects mode detected; skipping shared project preparation..."
else
  # Handle DELETE_ON_START at script level (before any sessions start)
  # This is done here (not per-session) to avoid race conditions
  if is_env_truthy "UNIFY_TESTS_DELETE_PROJ_ON_START"; then
    delete_shared_project "start"
  fi
  echo "Preparing shared project '${UNIFY_PROJECT:-UnityTests}'..."
  if [[ -f "$SCRIPT_DIR/_prepare_shared_project.py" ]]; then
    "$VENV_PY" "$SCRIPT_DIR/_prepare_shared_project.py"
  else
    echo "Warning: _prepare_shared_project.py not found." >&2
    echo "Falling back to random projects mode." >&2
    ENV_OVERRIDES+=( "UNIFY_TESTS_RAND_PROJ=True" "UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True" )
  fi
fi

# Build the command to run in each tmux session
run_cmd() {
  local target="$1"   # pytest target (file path or node id)
  local marker_arg="$2"  # optional marker filter (e.g., "-m eval")
  # Build the inner script first with safe %q for path/target, then quote the whole script with %q
  local inner
  local env_exports
  # Always export UTF-8 locale for proper emoji handling in session names
  # Enable cache stats tracking (UNILLM_CACHE_STATS must be set before importing unillm)
  env_exports='export LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 UNILLM_CACHE_STATS=true'
  if is_random_projects_mode; then
    # Random projects mode: each session gets its own isolated project.
    # Per-session deletion is safe here since projects don't overlap.
    env_exports="$env_exports UNIFY_TESTS_RAND_PROJ=True"
    # Pass delete flags only in random mode (safe per-session)
    if is_env_truthy "UNIFY_TESTS_DELETE_PROJ_ON_START"; then
      env_exports="$env_exports UNIFY_TESTS_DELETE_PROJ_ON_START=True"
    fi
    if is_env_truthy "UNIFY_TESTS_DELETE_PROJ_ON_EXIT"; then
      env_exports="$env_exports UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True"
    fi
  else
    # Shared project mode: skip session setup (already done by prepare script)
    env_exports="$env_exports UNIFY_SKIP_SESSION_SETUP=True"
  fi
  # Append user-provided --env overrides (includes UNITY_TEST_SOCKET for log scoping)
  local user_overrides
  user_overrides="$(build_env_exports)"
  if [[ -n "$user_overrides" ]]; then
    env_exports="$env_exports$user_overrides"
  fi
  # Build pytest command with optional marker filter, scenario overwrite, and extra args
  local pytest_cmd
  local extra_args=""
  if (( OVERWRITE_SCENARIOS )); then
    extra_args="--overwrite-scenarios"
  fi
  # Append any extra pytest args passed via --
  if (( ${#PYTEST_EXTRA_ARGS[@]} > 0 )); then
    for arg in "${PYTEST_EXTRA_ARGS[@]}"; do
      extra_args="$extra_args $(printf '%q' "$arg")"
    done
  fi
  if [[ -n "$marker_arg" ]]; then
    pytest_cmd=$(printf '%q -m pytest %s %s %q' "$VENV_PY" "$marker_arg" "$extra_args" "$target")
  else
    pytest_cmd=$(printf '%q -m pytest %s %q' "$VENV_PY" "$extra_args" "$target")
  fi
  # Build inner command with socket name directly interpolated (not via env var)
  # This ensures tmux commands target the correct isolated server
  # Note: LC_ALL=en_US.UTF-8 is required for Unicode emoji support in tmux session names
  # Note: Log paths are now auto-derived by conftest.py using UNITY_TEST_SOCKET + semantic naming
  # Inner command runs inside tmux session after pytest completes.
  # The rename-session uses "|| true" to gracefully handle race conditions
  # where multiple sessions complete simultaneously or external agents interfere.
  # The session ID is captured BEFORE pytest runs and exported as UNITY_TMUX_SESSION_ID
  # so pytest's conftest.py can write cache stats to a known temp file location.
  inner=$(printf '%s; export UNITY_TMUX_SESSION_ID=$(LC_ALL=en_US.UTF-8 tmux -L %q display-message -p -t "$TMUX_PANE" "#{session_id}"); cd %q && %s; status=$?; sname=$(LC_ALL=en_US.UTF-8 tmux -L %q display-message -p -t "$TMUX_PANE" "#{session_name}"); base="$sname"; case "$sname" in "p ✅ "*) base="${sname#p ✅ }" ;; "f ❌ "*) base="${sname#f ❌ }" ;; "r ⏳ "*) base="${sname#r ⏳ }" ;; esac; if [ $status -eq 0 ]; then pfx="p ✅"; else pfx="f ❌"; fi; LC_ALL=en_US.UTF-8 tmux -L %q rename-session -t "$sname" "$pfx $base" 2>/dev/null || true; if [ $status -eq 0 ]; then (sleep 10; LC_ALL=en_US.UTF-8 tmux -L %q kill-session -t "$UNITY_TMUX_SESSION_ID" 2>/dev/null; if ! LC_ALL=en_US.UTF-8 tmux -L %q ls >/dev/null 2>&1; then LC_ALL=en_US.UTF-8 tmux -L %q kill-server 2>/dev/null || true; fi) >/dev/null 2>&1 & disown; echo "All tests passed. This tmux session will close in 10s..."; fi; echo; echo "pytest exited with code: $status"; echo "(You are now in a shell. Press Ctrl-D to close this window.)"; exec bash -l' "$env_exports" "$TMUX_SOCKET" "$REPO_ROOT" "$pytest_cmd" "$TMUX_SOCKET" "$TMUX_SOCKET" "$TMUX_SOCKET" "$TMUX_SOCKET" "$TMUX_SOCKET")
  printf 'bash -lc %q' "$inner"
}

# Ensure we don't collide with existing sessions.
# Checks status prefix variants (r ⏳, p ✅, f ❌) since sessions get renamed
# after completion, which could cause race conditions with subsequent runs.
#
# When called WITHOUT a prefix: checks unprefixed AND all prefixed variants
# When called WITH a prefix: only checks prefixed variants (the unprefixed
#   session is the one we just created and are about to rename)
unique_session_name() {
  local input="$1" n=1

  # Strip any status prefix to get the base name for collision checking
  local base="$input"
  local prefix=""
  case "$input" in
    "r ⏳ "*) base="${input#r ⏳ }"; prefix="r ⏳ " ;;
    "p ✅ "*) base="${input#p ✅ }"; prefix="p ✅ " ;;
    "f ❌ "*) base="${input#f ❌ }"; prefix="f ❌ " ;;
  esac

  local candidate="$base"
  while true; do
    local found=0
    # If input had NO prefix, check unprefixed version too
    # (If it had a prefix, skip - that's our just-created session we're renaming)
    if [[ -z "$prefix" ]]; then
      tmux_cmd has-session -t "$candidate" 2>/dev/null && found=1
    fi
    # Always check all prefixed versions to detect renamed sessions
    tmux_cmd has-session -t "r ⏳ $candidate" 2>/dev/null && found=1
    tmux_cmd has-session -t "p ✅ $candidate" 2>/dev/null && found=1
    tmux_cmd has-session -t "f ❌ $candidate" 2>/dev/null && found=1

    if (( found == 0 )); then
      break
    fi
    ((n++))
    candidate="${base}-${n}"
  done

  # Return with original prefix (if any)
  printf "%s%s" "$prefix" "$candidate"
}

# Rename session with retry on "duplicate session" errors.
# Race conditions (e.g., multiple agents, fast-completing tests) can cause
# the unique_session_name check to pass but the rename to fail. This helper
# retries with incrementing suffixes until success or max retries.
rename_session_with_retry() {
  local sid="$1"
  local target_name="$2"
  local max_retries=3
  local attempt=0
  local current_name="$target_name"

  while (( attempt < max_retries )); do
    if tmux_cmd rename-session -t "$sid" "$current_name" 2>/dev/null; then
      printf "%s" "$current_name"
      return 0
    fi
    # Rename failed - likely duplicate. Add/increment suffix and retry.
    ((attempt++)) || true
    # Strip any existing retry suffix and add new one
    local base="${target_name%-dup[0-9]*}"
    current_name="${base}-dup${attempt}"
    sleep 0.1  # Brief delay before retry
  done

  # All retries failed - log warning but don't crash the script.
  # The session exists (with unprefixed name), tests will still run.
  echo "Warning: rename-session failed after $max_retries retries for $target_name" >&2
  # Return the session's current name (query it since rename failed)
  local actual_name
  actual_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "$target_name")
  printf "%s" "$actual_name"
  return 0  # Don't fail the script
}

# Count currently pending (running) sessions in our socket
count_pending_sessions() {
  local count=0
  while IFS= read -r name; do
    if [[ "$name" == "r"* ]]; then
      ((count++)) || true
    fi
  done < <(tmux_cmd list-sessions -F "#{session_name}" 2>/dev/null || true)
  echo "$count"
}

# Wait until we have fewer than MAX_JOBS pending sessions
# While waiting, report any sessions that have completed (inline pass/fail feedback)
wait_for_job_slot() {
  if (( MAX_JOBS == 0 )); then
    return 0  # No limit
  fi

  while true; do
    local pending
    pending=$(count_pending_sessions)
    if (( pending < MAX_JOBS )); then
      # Report any completions that freed up this slot BEFORE returning
      # (ensures pass/fail appears before the new pending session that fills this slot)
      report_completed_sessions
      return 0
    fi
    # Report completions while waiting for a slot
    report_completed_sessions
    sleep 0.5
  done
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
# find <roots> -mindepth 1 \( -type d \( -name EX1 -o EX2 ... \) -prune \) -o \( -type f -name "test_*.py" -print0 \)
# Note: -mindepth 1 ensures root directories aren't pruned even if they match EXCLUDE_DIRS
# (e.g., explicitly passing "fixtures/" should search it, not prune it)
build_find_cmd() {
  local -a cmd=( find )
  if (( ${#roots[@]} )); then
    cmd+=( "${roots[@]}" )
  else
    cmd+=( "." )
  fi

  # -mindepth 1: don't apply exclusions to root directories themselves
  cmd+=( -mindepth 1 "(" -type d "(" )
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

# Collect pytest node ids for multiple targets at once (batch collection)
# This is much faster than calling pytest --collect-only per file, as pytest
# initialization (~10s) happens only once instead of per file.
collect_nodes_batch() {
  local marker_arg="$1"  # optional marker filter (e.g., "-m eval")
  shift
  local targets=("$@")

  if (( ${#targets[@]} == 0 )); then
    return 0
  fi

  local cmd
  # Always use UNIFY_SKIP_SESSION_SETUP for collection - we only need test IDs,
  # not a real project. This avoids slow project creation/deletion per collection.
  local env_exports='export UNIFY_SKIP_SESSION_SETUP=True'
  # Append user-provided --env overrides
  local user_overrides
  user_overrides="$(build_env_exports)"
  if [[ -n "$user_overrides" ]]; then
    env_exports="$env_exports$user_overrides"
  fi

  # Build target list with proper quoting
  local quoted_targets=""
  for t in "${targets[@]}"; do
    quoted_targets+=" $(printf '%q' "$t")"
  done

  # Build collection filter args (marker filter + any -k/-m from PYTEST_EXTRA_ARGS)
  local collection_filters=""
  if [[ -n "$marker_arg" ]]; then
    collection_filters="$marker_arg"
  fi
  # Append collection-relevant args from PYTEST_EXTRA_ARGS (e.g., -k "pattern")
  if (( ${#PYTEST_COLLECTION_ARGS[@]} > 0 )); then
    for carg in "${PYTEST_COLLECTION_ARGS[@]}"; do
      collection_filters+=" $(printf '%q' "$carg")"
    done
  fi

  # Build collection command with filters
  if [[ -n "$collection_filters" ]]; then
    cmd=$(printf '%s; cd %q && %q -m pytest --collect-only -q %s %s' "$env_exports" "$REPO_ROOT" "$VENV_PY" "$collection_filters" "$quoted_targets")
  else
    cmd=$(printf '%s; cd %q && %q -m pytest --collect-only -q %s' "$env_exports" "$REPO_ROOT" "$VENV_PY" "$quoted_targets")
  fi
  # Remove color codes, keep only node ids (contain ::), ignore noise; never fail the script
  # Redirect stdin from /dev/null to prevent hangs when multiple processes compete for stdin
  # Note: stderr is captured to a temp file so collection errors can be surfaced if no tests found
  local stderr_file
  stderr_file=$(mktemp)
  local result
  result=$(bash -lc "$cmd" < /dev/null 2>"$stderr_file" | sed -E 's/\x1B\[[0-9;]*[mK]//g' | grep -E '::' || true)

  # If no tests collected and there was stderr output, show it for debugging
  if [[ -z "$result" && -s "$stderr_file" ]]; then
    echo "Warning: pytest collection produced no test nodes. stderr output:" >&2
    head -50 "$stderr_file" >&2
  fi
  rm -f "$stderr_file"

  echo "$result"
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
START_TIMES_FILE="$(mktemp)"
RESULTS_FILE="$(mktemp)"
trap 'rm -f "$tmp" "$START_TIMES_FILE" "$RESULTS_FILE"' EXIT
if (( ! SERIAL )); then
  # Default mode (per-test): expand directories/files into node ids using batch collection
  # Combine all targets for a single pytest --collect-only call (much faster)
  all_targets=()
  if (( ${#direct_files[@]} )); then
    all_targets+=( "${direct_files[@]}" )
  fi
  if (( ${#found_files[@]} )); then
    all_targets+=( "${found_files[@]}" )
  fi
  if (( ${#all_targets[@]} )); then
    while IFS= read -r nid; do
      [[ -n "$nid" ]] && printf '%s\0' "$nid" >> "$tmp"
    done < <(collect_nodes_batch "$MARKER_FILTER" "${all_targets[@]}")
  fi
  if (( ${#direct_nodes[@]} )); then
    printf '%s\0' "${direct_nodes[@]}" >> "$tmp"
  fi
elif [[ -n "$MARKER_FILTER" ]]; then
  # Default mode WITH marker filter: collect nodes first to find which files
  # have matching tests, then create one session per file (not per-node).
  # This prevents creating sessions for files with 0 matching tests.
  all_targets=()
  if (( ${#direct_files[@]} )); then
    all_targets+=( "${direct_files[@]}" )
  fi
  if (( ${#found_files[@]} )); then
    all_targets+=( "${found_files[@]}" )
  fi
  if (( ${#all_targets[@]} )); then
    # Collect node ids, extract unique file paths (bash 3.x compatible)
    # Use a temp file + sort -u to get unique file paths
    tmp_files="$(mktemp)"
    while IFS= read -r nid; do
      if [[ -n "$nid" && "$nid" == *"::"* ]]; then
        # Extract file path (everything before first ::)
        file_path="${nid%%::*}"
        echo "$file_path" >> "$tmp_files"
      fi
    done < <(collect_nodes_batch "$MARKER_FILTER" "${all_targets[@]}")
    # Output unique files that have matching tests
    while IFS= read -r file_path; do
      [[ -n "$file_path" ]] && printf '%s\0' "$file_path" >> "$tmp"
    done < <(sort -u "$tmp_files")
    rm -f "$tmp_files"
  fi
  if (( ${#direct_nodes[@]} )); then
    printf '%s\0' "${direct_nodes[@]}" >> "$tmp"
  fi
else
  # Default mode without marker filter: one session per file
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
  echo "Error: No tests found for the given path(s)." >&2
  echo "This could mean:" >&2
  echo "  - The path doesn't exist or contains no test_*.py files" >&2
  echo "  - pytest --collect-only failed (check for import errors)" >&2
  echo "  - A marker filter (--eval-only/--symbolic-only) excluded all tests" >&2
  exit 1
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

# Helper function to print log directory info (used at start and end)
print_log_directories() {
  echo "========================================================================"
  echo "📁 pytest logs:  logs/pytest/$LOG_SUBDIR/"
  echo "🔗 OTel traces:  logs/{unity|unify|unillm|orchestra}/ (per-repo), logs/all/ (cross-repo)"
  echo "📖 Logging docs: logs/README.md"
  echo "========================================================================"
}

# Print log directory info first (before session creation starts)
echo
print_log_directories
echo

if (( MAX_JOBS > 0 )); then
  echo "Concurrency limit: $MAX_JOBS simultaneous sessions"
else
  echo "Concurrency limit: unlimited"
fi

# Print header before drip-feeding session creation
echo "Creating ${#files[@]} tmux sessions..."

for target in "${files[@]}"; do
  # Report any completions before creating new sessions
  # (ensures passes/fails appear BEFORE the new pending sessions they freed slots for)
  report_completed_sessions

  # If job limit is set, wait for a slot before creating new session
  wait_for_job_slot

  base_sess="$(session_basename_for "$target")"
  session="$(unique_session_name "$base_sess")"

  # Window name = basename without .py
  fname="${target##*/}"
  wname="${fname%.py}"

  # Create the session first (no command), set remain-on-exit, then send the command.
  # Note: Log directory is created lazily by conftest.py only when a log file is
  # actually written, avoiding empty directories when sessions fail/are interrupted.
  # Note: Log paths are auto-derived by conftest.py (semantic name + timestamp in socket subdir)
  cmd="$(run_cmd "$target" "$MARKER_FILTER")"

  # Capture session ID to track this specific run robustly
  sid=$(tmux_cmd new-session -d -P -F "#{session_id}" -s "$session" -n "$wname" "$cmd")

  # Record start time for duration tracking
  echo "$sid $(date +%s)" >> "$START_TIMES_FILE"

  # Rename to pending state with retry logic for race conditions
  pending_name="$(unique_session_name "r ⏳ $session")"
  session="$(rename_session_with_retry "$sid" "$pending_name")"

  # Print session as it's created (drip-feed)
  echo "  - $session"

  made_sessions+=( "$session" )
  session_ids+=( "$sid" )
  # Track for cleanup on interrupt (SIGINT/SIGTERM)
  CREATED_SESSION_IDS+=( "$sid" )
done

# ---- Wait for all tests to complete ----
# Always block until completion (or timeout). Continue showing drip-feed of
# pass/fail results as tests complete.

# Count how many have already completed during session creation
completed_count=0
pending_count=0
for sid in "${session_ids[@]}"; do
  current_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
  if [[ "$current_name" == "r"* ]]; then
    ((pending_count++)) || true
  else
    ((completed_count++)) || true
  fi
done

# Print summary with completion status
if (( pending_count == 0 )); then
  echo "Created all ${#made_sessions[@]} tmux sessions. All completed!"
elif (( TIMEOUT > 0 )); then
  echo "Created all ${#made_sessions[@]} tmux sessions. $completed_count completed. Waiting for remaining $pending_count to complete (timeout: ${TIMEOUT}s)..."
else
  echo "Created all ${#made_sessions[@]} tmux sessions. $completed_count completed. Waiting for remaining $pending_count to complete..."
fi

wait_start=$(date +%s)
timed_out=0
while true; do
  # Report any newly completed sessions (drip-feed pass/fail inline)
  report_completed_sessions

  # Count remaining pending sessions
  pending_count=0
  for sid in "${session_ids[@]}"; do
    # Check name of our specific session IDs only
    current_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
    # Look for "r" prefix to detect pending state (r ⏳)
    if [[ "$current_name" == "r"* ]]; then
      ((pending_count++)) || true
    fi
  done

  if (( pending_count == 0 )); then
    # Final report to catch any last completions
    report_completed_sessions
    break
  fi

  # Check timeout if specified
  if (( TIMEOUT > 0 )); then
    elapsed=$(( $(date +%s) - wait_start ))
    if (( elapsed >= TIMEOUT )); then
      timed_out=1
      echo ""
      echo "Timeout reached after ${TIMEOUT}s. ${pending_count} session(s) still running."
      break
    fi
  fi

  sleep 1
done

if (( timed_out )); then
  echo "Tests did not complete within timeout. Check tmux sessions manually."
  exit 2
fi

echo ""
echo "All tests completed."

# Collect failures
declare -a failed_sessions=()
for sid in "${session_ids[@]}"; do
  current_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
  # Look for "f" prefix to detect failure (f ❌)
  if [[ "$current_name" == "f"* ]]; then
    failed_sessions+=( "$current_name" )
  fi
done

# Handle DELETE_ON_EXIT at script level (after all sessions complete)
# This is done here (not per-session) to avoid race conditions in shared mode
if ! is_random_projects_mode && is_env_truthy "UNIFY_TESTS_DELETE_PROJ_ON_EXIT"; then
  delete_shared_project "exit"
fi

# Print test stats (duration and cache) sorted fastest to slowest
echo ""
echo "========================================================================"
echo "TEST STATS: DURATION & CACHE (fastest → slowest)"
echo "========================================================================"

# Count passed and failed (use { grep || true; } to handle no-match case with pipefail)
pass_count=$( { grep '|pass|' "$RESULTS_FILE" || true; } 2>/dev/null | wc -l | tr -d ' ')
fail_count=$( { grep '|fail|' "$RESULTS_FILE" || true; } 2>/dev/null | wc -l | tr -d ' ')

# Build duration summary for both stdout and file output
DURATION_SUMMARY_FILE="$REPO_ROOT/logs/pytest/$LOG_SUBDIR/duration_summary.txt"
mkdir -p "$(dirname "$DURATION_SUMMARY_FILE")"

# Helper to print to both stdout and file
print_duration_line() {
  echo "$1"
  echo "$1" >> "$DURATION_SUMMARY_FILE"
}

# Helper to format cache hit rate as percentage string
format_cache_rate() {
  local hits="$1"
  local misses="$2"
  local total=$((hits + misses))
  if (( total == 0 )); then
    echo "  --%"
  else
    local pct=$((hits * 100 / total))
    printf "%3d%%" "$pct"
  fi
}

# Clear/create the summary file
> "$DURATION_SUMMARY_FILE"

# Print passed tests sorted by duration (fastest first, slowest last)
# Format: duration|status|hits|misses|name
if (( pass_count > 0 )); then
  echo ""
  print_duration_line "✅ PASSED ($pass_count tests):"
  print_duration_line "$(printf "  %6s  %6s  %s" "time" "cache" "test")"
  print_duration_line "$(printf "  %6s  %6s  %s" "----" "-----" "----")"
  { grep '|pass|' "$RESULTS_FILE" || true; } | sort -t'|' -k1 -n | while IFS='|' read -r dur status hits misses name; do
    cache_rate=$(format_cache_rate "$hits" "$misses")
    print_duration_line "$(printf "  %5ds  %6s  %s" "$dur" "$cache_rate" "$name")"
  done
fi

# Print failed tests sorted by duration (fastest first, slowest last)
if (( fail_count > 0 )); then
  print_duration_line ""
  print_duration_line "❌ FAILED ($fail_count tests):"
  print_duration_line "$(printf "  %6s  %6s  %s" "time" "cache" "test")"
  print_duration_line "$(printf "  %6s  %6s  %s" "----" "-----" "----")"
  { grep '|fail|' "$RESULTS_FILE" || true; } | sort -t'|' -k1 -n | while IFS='|' read -r dur status hits misses name; do
    cache_rate=$(format_cache_rate "$hits" "$misses")
    print_duration_line "$(printf "  %5ds  %6s  %s" "$dur" "$cache_rate" "$name")"
  done
fi

# Calculate and print aggregated totals
total_duration=0
total_hits=0
total_misses=0
while IFS='|' read -r dur status hits misses name; do
  total_duration=$((total_duration + dur))
  total_hits=$((total_hits + hits))
  total_misses=$((total_misses + misses))
done < "$RESULTS_FILE"

total_tests=$((pass_count + fail_count))
if (( total_tests > 0 )); then
  print_duration_line ""
  print_duration_line "========================================================================"
  print_duration_line "TOTALS ($total_tests tests)"
  print_duration_line "========================================================================"
  # Format duration as minutes:seconds if > 60s
  if (( total_duration >= 60 )); then
    mins=$((total_duration / 60))
    secs=$((total_duration % 60))
    duration_str="${mins}m ${secs}s"
  else
    duration_str="${total_duration}s"
  fi
  total_cache_rate=$(format_cache_rate "$total_hits" "$total_misses")
  total_calls=$((total_hits + total_misses))
  print_duration_line "  Serial duration: $duration_str"
  print_duration_line "  LLM cache: $total_cache_rate ($total_hits hits, $total_misses misses, $total_calls total)"
fi

echo ""
print_log_directories

if (( ${#failed_sessions[@]} > 0 )); then
  exit 1
else
  echo ""
  echo "All tests passed!"
  exit 0
fi
