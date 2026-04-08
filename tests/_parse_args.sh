#!/usr/bin/env bash
# Shared argument parsing for parallel test runners.
#
# This file is sourced by both parallel_run.sh and parallel_cloud_run.sh
# to ensure identical argument handling.
#
# Usage:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
#   source "$SCRIPT_DIR/_parse_args.sh"
#   parse_test_args "$@"
#
# After calling parse_test_args, these variables are populated:
#   SERIAL, TIMEOUT, NAME_PATTERN, EVAL_ONLY, SYMBOLIC_ONLY,
#   REPEAT_COUNT, OVERWRITE_SCENARIOS, MAX_JOBS, ENV_OVERRIDES[],
#   TAGS[], PYTEST_EXTRA_ARGS[], PYTEST_COLLECTION_ARGS[],
#   POSITIONAL_ARGS[]
#
# Additional functions:
#   resolve_test_paths REPO_ROOT   - Validates paths in POSITIONAL_ARGS, sets RESOLVED_TEST_PATHS[]
#   reconstruct_parallel_run_args  - Rebuilds args as string (for CI passthrough)
#   print_help                     - Prints usage (caller can override HELP_SCRIPT_NAME)

# ---- Detect CPU cores for default MAX_JOBS ----
if [[ "$(uname)" == "Darwin" ]]; then
  _PARSE_ARGS_NUM_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
else
  _PARSE_ARGS_NUM_CORES=$(nproc 2>/dev/null || echo 4)
fi

# Export for use in help text
DETECTED_CPU_CORES=$_PARSE_ARGS_NUM_CORES

# ---- Main argument parsing function ----
# Parses command line arguments and populates global variables.
# Returns 0 on success, 1 if help was requested, 2 on error.
parse_test_args() {
  # Initialize/reset all variables
  SERIAL=0
  TIMEOUT=0
  NAME_PATTERN=""
  EVAL_ONLY=0
  SYMBOLIC_ONLY=0
  REPEAT_COUNT=1
  OVERWRITE_SCENARIOS=0
  MAX_JOBS=$_PARSE_ARGS_NUM_CORES
  ENV_OVERRIDES=()
  TAGS=()
  PYTEST_EXTRA_ARGS=()
  PYTEST_COLLECTION_ARGS=()
  PYTEST_IGNORE_PATHS=()
  POSITIONAL_ARGS=()

  while (( "$#" )); do
    case "$1" in
      -t|--timeout)
        if [[ -n "${2-}" && "$2" =~ ^[0-9]+$ && "$2" -ge 1 ]]; then
          TIMEOUT="$2"
          shift 2
        else
          echo "Error: --timeout requires a positive integer (seconds)." >&2
          return 2
        fi
        ;;
      -s|--serial)
        SERIAL=1
        shift
        ;;
      -m|--match)
        if [[ -n "${2-}" ]]; then
          NAME_PATTERN="$2"
          shift 2
        else
          echo "Error: -m|--match requires a pattern argument (e.g., \"*_tool_docstring*\")." >&2
          return 2
        fi
        ;;
      -e|--env)
        if [[ -n "${2-}" && "$2" == *=* ]]; then
          ENV_OVERRIDES+=( "$2" )
          shift 2
        else
          echo "Error: -e|--env requires KEY=VALUE argument (e.g., --env UNILLM_CACHE=false)." >&2
          return 2
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
          return 2
        fi
        ;;
      --overwrite-scenarios)
        OVERWRITE_SCENARIOS=1
        shift
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
          return 2
        fi
        ;;
      -j|--jobs)
        if [[ -z "${2-}" ]]; then
          echo "Error: -j|--jobs requires an argument (e.g., --jobs 8, --jobs 0, --jobs none)." >&2
          return 2
        fi
        # Accept positive integers, 0, or keywords for unlimited
        local arg_lower
        arg_lower=$(echo "$2" | tr '[:upper:]' '[:lower:]')
        if [[ "$2" =~ ^[0-9]+$ ]]; then
          MAX_JOBS="$2"
        elif [[ "$arg_lower" == "none" || "$arg_lower" == "unlimited" || "$arg_lower" == "inf" ]]; then
          MAX_JOBS=0
        else
          echo "Error: -j|--jobs requires a non-negative integer or 'none'/'unlimited' (e.g., --jobs 8, --jobs 0, --jobs none)." >&2
          return 2
        fi
        shift 2
        ;;
      -h|--help)
        # Return 1 to signal help was requested; caller should print help and exit
        return 1
        ;;
      --)
        shift
        PYTEST_EXTRA_ARGS=("$@")
        # Extract collection-relevant args for use during test discovery.
        # These affect which tests are collected, not just how they run:
        #   -k/-m: filter by keyword/marker expression
        #   --ignore/--ignore-glob: exclude paths from collection
        local _coll_i=0
        while (( _coll_i < ${#PYTEST_EXTRA_ARGS[@]} )); do
          local _coll_arg="${PYTEST_EXTRA_ARGS[_coll_i]}"
          case "$_coll_arg" in
            -k|-m)
              # Next arg is the value (e.g., -k "pattern")
              if (( _coll_i + 1 < ${#PYTEST_EXTRA_ARGS[@]} )); then
                PYTEST_COLLECTION_ARGS+=( "$_coll_arg" "${PYTEST_EXTRA_ARGS[_coll_i+1]}" )
                ((_coll_i+=2))
              else
                ((_coll_i++))
              fi
              ;;
            --ignore)
              # Next arg is the path (e.g., --ignore tests/actor)
              if (( _coll_i + 1 < ${#PYTEST_EXTRA_ARGS[@]} )); then
                PYTEST_COLLECTION_ARGS+=( "$_coll_arg" "${PYTEST_EXTRA_ARGS[_coll_i+1]}" )
                PYTEST_IGNORE_PATHS+=( "${PYTEST_EXTRA_ARGS[_coll_i+1]%/}" )
                ((_coll_i+=2))
              else
                ((_coll_i++))
              fi
              ;;
            -k=*|-m=*)
              # Value is attached (e.g., -k="pattern")
              PYTEST_COLLECTION_ARGS+=( "$_coll_arg" )
              ((_coll_i++))
              ;;
            --ignore=*)
              # Value is attached (e.g., --ignore=tests/actor)
              PYTEST_COLLECTION_ARGS+=( "$_coll_arg" )
              local _ival="${_coll_arg#--ignore=}"
              PYTEST_IGNORE_PATHS+=( "${_ival%/}" )
              ((_coll_i++))
              ;;
            --ignore-glob=*)
              PYTEST_COLLECTION_ARGS+=( "$_coll_arg" )
              ((_coll_i++))
              ;;
            --keyword=*|--markers=*)
              # Long form with attached value
              PYTEST_COLLECTION_ARGS+=( "$_coll_arg" )
              ((_coll_i++))
              ;;
            *)
              ((_coll_i++))
              ;;
          esac
        done
        break
        ;;
      -*)
        echo "Error: Unknown option: $1" >&2
        echo "To pass pytest options (like -k, -v, -x), use -- before them:" >&2
        echo "  Example: parallel_run.sh tests/foo.py -- -k 'pattern'" >&2
        echo "Run with -h for all options." >&2
        return 2
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
    return 2
  fi

  return 0
}

# ---- Resolve and validate test paths ----
# Takes repo root as argument, reads from POSITIONAL_ARGS, sets RESOLVED_TEST_PATHS.
# Returns 0 on success, 1 if any path not found.
resolve_test_paths() {
  local repo_root="$1"
  RESOLVED_TEST_PATHS=()

  for path in "${POSITIONAL_ARGS[@]}"; do
    if [[ -e "$repo_root/$path" ]]; then
      RESOLVED_TEST_PATHS+=("$path")
    elif [[ -e "$repo_root/tests/$path" ]]; then
      RESOLVED_TEST_PATHS+=("tests/$path")
    else
      echo "Error: Path not found: $path" >&2
      echo "  Also tried: tests/$path" >&2
      echo "  (paths are relative to repo root: $repo_root)" >&2
      return 1
    fi
  done

  return 0
}

# ---- Reconstruct flags as a string ----
# Used by parallel_cloud_run.sh to rebuild args for CI passthrough.
# Does NOT include test paths (those are handled separately).
# Optional argument: "include-env" to include --env flags in output.
reconstruct_parallel_run_args() {
  local include_env=0
  [[ "${1:-}" == "include-env" ]] && include_env=1

  local args=""

  (( SERIAL )) && args="$args -s"
  (( TIMEOUT > 0 )) && args="$args --timeout $TIMEOUT"
  [[ -n "$NAME_PATTERN" ]] && args="$args -m $(printf '%q' "$NAME_PATTERN")"
  (( EVAL_ONLY )) && args="$args --eval-only"
  (( SYMBOLIC_ONLY )) && args="$args --symbolic-only"
  (( REPEAT_COUNT > 1 )) && args="$args --repeat $REPEAT_COUNT"
  (( OVERWRITE_SCENARIOS )) && args="$args --overwrite-scenarios"
  # Note: MAX_JOBS is not passed to CI (CI has its own resource limits)

  for tag in "${TAGS[@]}"; do
    args="$args --tags $(printf '%q' "$tag")"
  done

  # Include --env flags if requested
  if (( include_env )); then
    for kv in "${ENV_OVERRIDES[@]}"; do
      args="$args --env $(printf '%q' "$kv")"
    done
  fi

  if (( ${#PYTEST_EXTRA_ARGS[@]} > 0 )); then
    args="$args --"
    for arg in "${PYTEST_EXTRA_ARGS[@]}"; do
      args="$args $(printf '%q' "$arg")"
    done
  fi

  # Trim leading space
  echo "${args# }"
}

# ---- Print help text ----
# Caller can set HELP_SCRIPT_NAME before calling to customize the script name shown.
# Caller can set HELP_EXTRA_OPTIONS to add script-specific options to the help.
print_help() {
  local script_name="${HELP_SCRIPT_NAME:-parallel_run.sh}"
  cat << EOF
Usage: $script_name [options] [targets...]

Run pytest tests in parallel tmux sessions.
Always blocks until all tests complete (or timeout).

Options:
  -t, --timeout N      Abort if tests don't complete within N seconds
  -s, --serial         One session per file (default: one per test)
  -m, --match PATTERN  Filter files by glob pattern
  -e, --env KEY=VALUE  Set environment variable (repeatable)
  -j, --jobs N         Max concurrent sessions (default: CPU cores, currently $DETECTED_CPU_CORES)
  --eval-only          Run only @pytest.mark.eval tests
  --symbolic-only      Run only non-eval tests
  --repeat N           Run each test N times
  --tags TAG           Tag runs for filtering (repeatable)
  --overwrite-scenarios  Delete and recreate test scenarios
  -h, --help           Show this help
  --                   Pass remaining args directly to pytest
${HELP_EXTRA_OPTIONS:-}
Examples:
  $script_name tests/                    # Run all tests
  $script_name tests/foo.py             # Run one file
  $script_name --timeout 300 tests/     # 5-minute timeout
  $script_name -s tests/                # Serial mode (per-file)
  $script_name -j 8 tests/              # Limit to 8 concurrent
  $script_name --eval-only tests/       # Only eval tests
  $script_name -e UNILLM_CACHE=false tests/
  $script_name tests/ -- -v --tb=short  # Pass args to pytest
  $script_name tests/ -- -k 'gpt-5'     # Filter by test name pattern
EOF
}
