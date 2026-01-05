#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# parallel_queries.sh - Run repairs queries in parallel tmux sessions
#
# Similar to tests/parallel_run.sh but for repairs metrics queries.
#
# Usage:
#   ./parallel_queries.sh --all                    # All queries, default params
#   ./parallel_queries.sh --all --expand-params    # All queries × all param combos
#   ./parallel_queries.sh --all --full-matrix      # Full matrix: all queries × all params, nested dirs
#   ./parallel_queries.sh --query jobs_completed_per_day
#   ./parallel_queries.sh --query first_time_fix_rate --query no_access_rate
#   ./parallel_queries.sh --all -j 5               # Limit to 5 concurrent sessions
#   ./parallel_queries.sh --all -w                 # Wait for completion
# ==============================================================================

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# Source common utilities
source "$SCRIPT_DIR/_repairs_common.sh"

# Resolve repo root (3 levels up from intranet/repairs_agent/scripts/)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"

# Load .env if exists
_ENV_FILE="$REPO_ROOT/.env"
if [[ -f "$_ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$_ENV_FILE"
  set +a
fi
unset _ENV_FILE

# Increase file descriptor limit for parallel connections
ulimit -n 4096 2>/dev/null || true

TMUX_SOCKET="$REPAIRS_TMUX_SOCKET"
LOG_SUBDIR="${REPAIRS_LOG_SUBDIR:-$(_derive_log_subdir "$TMUX_SOCKET")}"

# Wrapper for all tmux commands (uses locale from _repairs_common.sh)
tmux_cmd() {
  tmux -L "$TMUX_SOCKET" "$@"
}

# ==============================================================================
# Session tracking and cleanup
# ==============================================================================
declare -a CREATED_SESSION_IDS=()
REPORTED_COMPLETIONS=""

_is_reported() {
  local sid="$1"
  [[ "$REPORTED_COMPLETIONS" == *"${sid}"* ]]
}

_mark_reported() {
  local sid="$1"
  REPORTED_COMPLETIONS="${REPORTED_COMPLETIONS}${sid}:"
}

report_completed_sessions() {
  (( ${#CREATED_SESSION_IDS[@]} == 0 )) && return 0

  for sid in "${CREATED_SESSION_IDS[@]}"; do
    _is_reported "$sid" && continue

    local current_name
    current_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
    [[ -z "$current_name" ]] && continue

    case "$current_name" in
      "p ✅ "*)
        local base="${current_name#p ✅ }"
        print_pass "$base"
        _mark_reported "$sid"
        ;;
      "f ❌ "*)
        local base="${current_name#f ❌ }"
        print_fail "$base"
        _mark_reported "$sid"
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
      tmux_cmd list-panes -t "$sid" -F '#{pane_pid}' 2>/dev/null | while read -r pid; do
        if [[ -n "$pid" ]]; then
          kill -TERM "-$pid" 2>/dev/null || true
        fi
      done
      sleep 0.1
      tmux_cmd kill-session -t "$sid" 2>/dev/null || true
    done
    echo "Cleanup complete."
  fi
}

trap '_cleanup_sessions INT; exit 130' INT
trap '_cleanup_sessions TERM; exit 143' TERM

# ==============================================================================
# Configuration
# ==============================================================================
WAIT_FOR_COMPLETION=0
WAIT_TIMEOUT=0
MAX_JOBS=10
EXPAND_PARAMS=0
FULL_MATRIX=0
RUN_ALL=0
PROJECT="RepairsAgent"
VERBOSE=0
DEBUG=0
NO_LOG=0
LOG_DIR=""
INCLUDE_PLOTS=0

declare -a QUERY_IDS=()

# ==============================================================================
# Parse arguments
# ==============================================================================
while (( "$#" )); do
  case "$1" in
    -w|--wait)
      WAIT_FOR_COMPLETION=1
      if [[ -n "${2-}" && "$2" =~ ^[0-9]+$ && "$2" -ge 1 ]]; then
        WAIT_TIMEOUT="$2"
        shift 2
      else
        shift
      fi
      ;;
    -j|--jobs)
      if [[ -z "${2-}" ]]; then
        error "-j|--jobs requires an argument"
        exit 2
      fi
      if [[ "$2" =~ ^[0-9]+$ ]]; then
        MAX_JOBS="$2"
      else
        error "-j|--jobs requires a positive integer"
        exit 2
      fi
      shift 2
      ;;
    --all)
      RUN_ALL=1
      shift
      ;;
    --expand-params)
      EXPAND_PARAMS=1
      shift
      ;;
    --full-matrix)
      FULL_MATRIX=1
      shift
      ;;
    --query|-q)
      if [[ -n "${2-}" ]]; then
        QUERY_IDS+=( "$2" )
        shift 2
      else
        error "--query requires a query ID"
        exit 2
      fi
      ;;
    --project)
      if [[ -n "${2-}" ]]; then
        PROJECT="$2"
        shift 2
      else
        error "--project requires a value"
        exit 2
      fi
      ;;
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    --no-log)
      NO_LOG=1
      shift
      ;;
    --log-dir)
      if [[ -n "${2-}" ]]; then
        LOG_DIR="$2"
        shift 2
      else
        error "--log-dir requires a path"
        exit 2
      fi
      ;;
    --include-plots)
      INCLUDE_PLOTS=1
      shift
      ;;
    -h|--help)
      cat << EOF
Usage: parallel_queries.sh [options]

Run repairs queries in parallel tmux sessions.

Query Selection:
  --all                 Run all available queries
  --query, -q ID        Run specific query (can be repeated)
  --expand-params       Expand all parameter combinations per query (flat structure)
  --full-matrix         Full matrix: all queries × all params with nested directories
                        Creates subdirectories per metric with per-metric summaries

Execution Control:
  -j, --jobs N          Max concurrent sessions (default: 10)
  -w, --wait [N]        Wait for completion (optional timeout in seconds)

Script Options:
  --project NAME        Project context (default: RepairsAgent)
  -v, --verbose         Enable verbose logging in queries
  --debug               Enable debug logging in queries
  --no-log              Disable file logging in queries
  --log-dir PATH        Custom log directory for queries
  --include-plots       Generate visualization URLs for query results

General:
  -h, --help            Show this help

Examples:
  # Run all queries with default parameters
  ./parallel_queries.sh --all

  # Run all queries with all parameter variations (flat log structure)
  ./parallel_queries.sh --all --expand-params

  # Full matrix with nested directories per metric
  ./parallel_queries.sh --all --full-matrix -j 8 -w

  # Run specific queries
  ./parallel_queries.sh --query jobs_completed_per_day --query no_access_rate

  # Limit concurrency and wait for completion
  ./parallel_queries.sh --all -j 5 -w

  # Run with verbose output
  ./parallel_queries.sh --all --verbose

  # Wait with timeout
  ./parallel_queries.sh --all -w 300

  # Run with plot generation
  ./parallel_queries.sh --all --include-plots -w
EOF
      exit 0
      ;;
    *)
      error "Unknown option: $1"
      exit 2
      ;;
  esac
done

# ==============================================================================
# Validation
# ==============================================================================
if (( ! RUN_ALL && ${#QUERY_IDS[@]} == 0 )); then
  error "Either --all or at least one --query is required"
  echo ""
  echo "Try: $0 --help"
  exit 1
fi

# ==============================================================================
# Python environment
# ==============================================================================
VENV_DIR="$REPO_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  error "Python venv not found at $VENV_PY"
  error "Run 'uv sync --all-groups' from repo root first"
  exit 1
fi

# ==============================================================================
# Generate query specifications
# ==============================================================================
GENERATOR="$SCRIPT_DIR/_query_generator.py"

if [[ ! -f "$GENERATOR" ]]; then
  error "Query generator not found: $GENERATOR"
  exit 1
fi

# Build generator arguments
gen_args=()
if (( RUN_ALL )); then
  gen_args+=( "--all" )
else
  for qid in "${QUERY_IDS[@]}"; do
    gen_args+=( "--query" "$qid" )
  done
fi
if (( FULL_MATRIX )); then
  gen_args+=( "--full-matrix" )
elif (( EXPAND_PARAMS )); then
  gen_args+=( "--expand-params" )
fi
gen_args+=( "--format" "jsonl" )

# Generate specs
QUERY_SPECS=$("$VENV_PY" "$GENERATOR" "${gen_args[@]}")

if [[ -z "$QUERY_SPECS" ]]; then
  error "No queries generated. Check your query IDs."
  exit 1
fi

# Count queries
QUERY_COUNT=$(echo "$QUERY_SPECS" | wc -l | tr -d ' ')

# ==============================================================================
# Session management functions
# ==============================================================================
unique_session_name() {
  local input="$1" n=1

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
    if [[ -z "$prefix" ]]; then
      tmux_cmd has-session -t "$candidate" 2>/dev/null && found=1
    fi
    tmux_cmd has-session -t "r ⏳ $candidate" 2>/dev/null && found=1
    tmux_cmd has-session -t "p ✅ $candidate" 2>/dev/null && found=1
    tmux_cmd has-session -t "f ❌ $candidate" 2>/dev/null && found=1

    if (( found == 0 )); then
      break
    fi
    ((n++))
    candidate="${base}-${n}"
  done

  printf "%s%s" "$prefix" "$candidate"
}

count_pending_sessions() {
  local count=0
  while IFS= read -r name; do
    if [[ "$name" == "r"* ]]; then
      ((count++)) || true
    fi
  done < <(tmux_cmd list-sessions -F "#{session_name}" 2>/dev/null || true)
  echo "$count"
}

wait_for_job_slot() {
  if (( MAX_JOBS == 0 )); then
    return 0
  fi

  while true; do
    local pending
    pending=$(count_pending_sessions)
    if (( pending < MAX_JOBS )); then
      report_completed_sessions
      return 0
    fi
    report_completed_sessions
    sleep 0.5
  done
}

# ==============================================================================
# Build run command
# ==============================================================================
build_run_cmd() {
  local query_id="$1"
  local params_json="$2"
  local metric_subdir="${3:-}"

  # Build the Python command
  local py_script="$SCRIPT_DIR/run_repairs_query.py"
  local py_cmd="$VENV_PY $py_script --query $query_id"

  # Add params if not empty
  if [[ "$params_json" != "{}" && -n "$params_json" ]]; then
    py_cmd="$py_cmd --params '$params_json'"
  fi

  # Add flags
  py_cmd="$py_cmd --project $PROJECT"

  # Always pass log-subdir for per-terminal isolation (like tests/parallel_run.sh)
  py_cmd="$py_cmd --log-subdir '$LOG_SUBDIR'"

  # In full-matrix mode, pass metric subdirectory for nested structure
  # and skip per-query summaries (shell script will generate them)
  if [[ -n "$metric_subdir" ]]; then
    py_cmd="$py_cmd --metric-subdir '$metric_subdir' --no-summary"
  fi

  if (( VERBOSE )); then
    py_cmd="$py_cmd --verbose"
  fi
  if (( DEBUG )); then
    py_cmd="$py_cmd --debug"
  fi
  if (( NO_LOG )); then
    py_cmd="$py_cmd --no-log"
  fi
  if [[ -n "$LOG_DIR" ]]; then
    py_cmd="$py_cmd --log-dir '$LOG_DIR'"
  fi
  if (( INCLUDE_PLOTS )); then
    py_cmd="$py_cmd --include-plots"
  fi

  # Build the inner script with status detection
  # Note: Use C.UTF-8 as fallback if en_US.UTF-8 not available
  local inner
  inner=$(cat << 'INNEREOF'
# Set UTF-8 locale (with fallback)
if locale -a 2>/dev/null | grep -qi 'en_US.utf'; then
  export LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
elif locale -a 2>/dev/null | grep -qi 'C.utf'; then
  export LC_ALL=C.UTF-8 LANG=C.UTF-8
fi
cd REPO_ROOT_PLACEHOLDER
PYCMD_PLACEHOLDER
status=$?
sname=$(tmux -L SOCKET_PLACEHOLDER display-message -p -t "$TMUX_PANE" "#{session_name}" 2>/dev/null || echo "")
base="$sname"
case "$sname" in
  "p ✅ "*) base="${sname#p ✅ }" ;;
  "f ❌ "*) base="${sname#f ❌ }" ;;
  "r ⏳ "*) base="${sname#r ⏳ }" ;;
esac
if [ $status -eq 0 ]; then
  pfx="p ✅"
else
  pfx="f ❌"
fi
tmux -L SOCKET_PLACEHOLDER rename-session -t "$sname" "$pfx $base" 2>/dev/null || true
if [ $status -eq 0 ]; then
  sid=$(tmux -L SOCKET_PLACEHOLDER display-message -p -t "$TMUX_PANE" "#{session_id}" 2>/dev/null || echo "")
  if [ -n "$sid" ]; then
    (sleep 10; tmux -L SOCKET_PLACEHOLDER kill-session -t "$sid" 2>/dev/null; if ! tmux -L SOCKET_PLACEHOLDER ls >/dev/null 2>&1; then tmux -L SOCKET_PLACEHOLDER kill-server 2>/dev/null || true; fi) >/dev/null 2>&1 & disown
  fi
  echo "Query completed successfully. This tmux session will close in 10s..."
fi
echo
echo "Query exited with code: $status"
echo "(You are now in a shell. Press Ctrl-D to close this window.)"
exec bash -l
INNEREOF
)

  # Replace placeholders
  inner="${inner//REPO_ROOT_PLACEHOLDER/$REPO_ROOT}"
  inner="${inner//PYCMD_PLACEHOLDER/$py_cmd}"
  inner="${inner//SOCKET_PLACEHOLDER/$TMUX_SOCKET}"

  printf 'bash -lc %q' "$inner"
}

# ==============================================================================
# Main execution
# ==============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║         Midland Heart Repairs - Parallel Query Runner              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

info "Queries to run: $QUERY_COUNT"
info "Max concurrent: $MAX_JOBS"
info "Tmux socket: $TMUX_SOCKET"
if (( FULL_MATRIX )); then
  info "Mode: Full matrix (all queries × all params, nested dirs)"
elif (( EXPAND_PARAMS )); then
  info "Mode: Expanded parameters (all combinations, flat structure)"
else
  info "Mode: Default parameters"
fi
echo ""

declare -a made_sessions=()
declare -a session_ids=()

echo "Creating tmux sessions:"

while IFS= read -r spec_line; do
  [[ -z "$spec_line" ]] && continue

  # Parse JSON spec
  query_id=$(echo "$spec_line" | "$VENV_PY" -c "import sys, json; d=json.loads(sys.stdin.read()); print(d['query_id'])")
  params_json=$(echo "$spec_line" | "$VENV_PY" -c "import sys, json; d=json.loads(sys.stdin.read()); print(json.dumps(d['params']))")
  session_name=$(echo "$spec_line" | "$VENV_PY" -c "import sys, json; d=json.loads(sys.stdin.read()); print(d['session_name'])")
  # In full-matrix mode, extract metric_subdir (may be empty in non-full-matrix mode)
  metric_subdir=$(echo "$spec_line" | "$VENV_PY" -c "import sys, json; d=json.loads(sys.stdin.read()); print(d.get('metric_subdir', ''))")

  # Report completions and wait for slot
  report_completed_sessions
  wait_for_job_slot

  # Get unique session name
  session="$(unique_session_name "$session_name")"
  wname="${query_id}"

  # Build command
  cmd="$(build_run_cmd "$query_id" "$params_json" "$metric_subdir")"

  # Create session
  sid=$(tmux_cmd new-session -d -P -F "#{session_id}" -s "$session" -n "$wname" "$cmd")

  # Rename to pending
  pending_name="$(unique_session_name "r ⏳ $session")"
  tmux_cmd rename-session -t "$sid" "$pending_name"
  session="$pending_name"

  echo "  - $session"

  made_sessions+=( "$session" )
  session_ids+=( "$sid" )
  CREATED_SESSION_IDS+=( "$sid" )

done <<< "$QUERY_SPECS"

echo ""
success "Created ${#made_sessions[@]} tmux sessions"

echo ""
echo "========================================================================"
echo "📁 Query logs: .repairs_queries/$LOG_SUBDIR/"
echo "========================================================================"
echo ""
echo "Commands:"
echo "  • List sessions:   tmux -L $TMUX_SOCKET ls"
echo "  • Attach:          tmux -L $TMUX_SOCKET attach -t <session>"
echo "  • Kill all:        tmux -L $TMUX_SOCKET kill-server"
echo ""

# ==============================================================================
# Wait for completion
# ==============================================================================
if (( WAIT_FOR_COMPLETION )); then
  if (( WAIT_TIMEOUT > 0 )); then
    echo "Waiting for queries to complete (timeout: ${WAIT_TIMEOUT}s)..."
  else
    echo "Waiting for queries to complete..."
  fi
  echo ""

  wait_start=$(date +%s)
  timed_out=0

  while true; do
    pending_count=0
    for sid in "${session_ids[@]}"; do
      current_name=$(tmux_cmd display-message -p -t "$sid" "#{session_name}" 2>/dev/null || echo "")
      if [[ "$current_name" == "r"* ]]; then
        ((pending_count++)) || true
      fi
    done

    # Report any completions during wait
    report_completed_sessions

    if (( pending_count == 0 )); then
      break
    fi

    if (( WAIT_TIMEOUT > 0 )); then
      elapsed=$(( $(date +%s) - wait_start ))
      if (( elapsed >= WAIT_TIMEOUT )); then
        timed_out=1
        warn "Timeout reached after ${WAIT_TIMEOUT}s. ${pending_count} session(s) still running."
        break
      fi
    fi

    sleep 1
  done

  if (( timed_out )); then
    error "Queries did not complete within timeout. Check tmux sessions manually."
    exit 2
  fi

  # Generate aggregate summary for all logs in this run FIRST
  # (so we can read stats from it for terminal output)
  echo ""
  echo "Generating aggregate summary..."
  RUN_LOG_DIR="$REPO_ROOT/.repairs_queries/$LOG_SUBDIR"
  if [[ -d "$RUN_LOG_DIR" ]]; then
    if (( FULL_MATRIX )); then
      # In full-matrix mode, generate per-metric summaries first
      echo "  Generating per-metric summaries..."
      for metric_dir in "$RUN_LOG_DIR"/*/; do
        if [[ -d "$metric_dir" ]]; then
          metric_name=$(basename "$metric_dir")
          "$VENV_PY" "$SCRIPT_DIR/query_logger.py" "$metric_dir" --per-metric 2>/dev/null || true
        fi
      done
      # Then generate global summary
      "$VENV_PY" "$SCRIPT_DIR/query_logger.py" "$RUN_LOG_DIR" --full-matrix 2>/dev/null || true
    else
      "$VENV_PY" "$SCRIPT_DIR/query_logger.py" "$RUN_LOG_DIR" 2>/dev/null || true
    fi
  fi

  # Read stats from the generated summary file (more reliable than checking tmux sessions)
  SUMMARY_FILE="$RUN_LOG_DIR/_run_summary.log"
  passed=0
  failed=0
  total=0

  if [[ -f "$SUMMARY_FILE" ]]; then
    # Parse stats from summary file
    total=$(grep -oP 'Total Queries:\s+\K\d+' "$SUMMARY_FILE" 2>/dev/null || echo "0")
    passed=$(grep -oP 'Total Successful:\s+\K\d+' "$SUMMARY_FILE" 2>/dev/null || \
             grep -oP 'Successful:\s+\K\d+' "$SUMMARY_FILE" 2>/dev/null || echo "0")
    failed=$(grep -oP 'Total Failed:\s+\K\d+' "$SUMMARY_FILE" 2>/dev/null || \
             grep -oP 'Failed:\s+\K\d+' "$SUMMARY_FILE" 2>/dev/null || echo "0")
  fi

  # Fallback: count log files if summary parsing failed
  if (( total == 0 )); then
    if (( FULL_MATRIX )); then
      total=$(find "$RUN_LOG_DIR" -name "*.log" ! -name "_*" 2>/dev/null | wc -l)
    else
      total=$(find "$RUN_LOG_DIR" -maxdepth 1 -name "*.log" ! -name "_*" 2>/dev/null | wc -l)
    fi
    # Assume all passed if we can't determine (better than showing 0)
    passed=$total
    failed=0
  fi

  # Final summary output
  echo ""
  echo "========================================================================"
  echo "                           FINAL RESULTS"
  echo "========================================================================"
  echo ""
  echo "  Total Queries:   $total"
  echo "  Successful:      $passed"
  echo "  Failed:          $failed"
  echo ""
  echo "📄 Summary: $SUMMARY_FILE"
  echo ""
  echo "========================================================================"
  if (( failed > 0 )); then
    echo -e "${RED}${BOLD}❌ ${failed}/${total} queries failed${NC}"
    echo ""
    error "Some queries failed. Check logs in .repairs_queries/$LOG_SUBDIR/"
    echo ""
    echo "To inspect failures:"
    echo "  • View summary:     cat .repairs_queries/$LOG_SUBDIR/_run_summary.log"
    if (( FULL_MATRIX )); then
      echo "  • Per-metric logs:  ls .repairs_queries/$LOG_SUBDIR/*/"
    fi
    echo "  • List sessions:    tmux -L $TMUX_SOCKET ls"
    echo "  • Attach to failed: tmux -L $TMUX_SOCKET attach -t '<session>'"
    exit 1
  else
    echo -e "${GREEN}${BOLD}✅ All ${total} queries passed!${NC}"
    echo "========================================================================"
    exit 0
  fi
fi
