#!/usr/bin/env bash
set -euo pipefail

# Grid Search Runner
# ==================
# Runs tests across all combinations of settings values.
#
# Usage:
#   ./.grid_search.sh --env KEY=val1|val2 --env KEY2=a|b|c [options] [targets...]
#
# The pipe character "|" separates multiple values for a setting.
# A full Cartesian product (grid) of all combinations is generated.
#
# Example:
#   ./.grid_search.sh --env UNIFY_MODEL="gpt-4o|claude-3" --env UNIFY_CACHE="true|false" tests/
#
# Generates 4 runs:
#   1. UNIFY_MODEL=gpt-4o UNIFY_CACHE=true
#   2. UNIFY_MODEL=gpt-4o UNIFY_CACHE=false
#   3. UNIFY_MODEL=claude-3 UNIFY_CACHE=true
#   4. UNIFY_MODEL=claude-3 UNIFY_CACHE=false

# Resolve script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PARALLEL_RUN="$SCRIPT_DIR/.parallel_run.sh"

# Check that .parallel_run.sh exists
if [[ ! -x "$PARALLEL_RUN" ]]; then
  echo "Error: .parallel_run.sh not found or not executable at $PARALLEL_RUN" >&2
  exit 1
fi

# Arrays to collect grid variables and pass-through options
declare -a GRID_VARS=()      # KEY=val1|val2|val3 entries
declare -a PASSTHROUGH=()    # Other --env entries (no pipe) and all other args
declare -a TARGETS=()        # Test targets (files/directories)

DRY_RUN=0
WAIT_FOR_ALL=0

usage() {
  cat <<'USAGE'
Grid Search Runner
==================

Run tests across all combinations of settings values.

Usage:
  ./.grid_search.sh [options] --env KEY=val1|val2 [--env KEY2=a|b] [targets...]

Options:
  --env KEY=val1|val2   Specify multiple values for a setting (pipe-separated)
  --env KEY=value       Single value (passed through to all runs)
  -n, --dry-run         Show generated commands without executing
  --wait-all            Wait for all grid runs to complete (runs sequentially with --wait)
  -h, --help            Show this help

All other options and arguments are passed through to .parallel_run.sh.

Examples:
  # Grid search across models and cache settings
  ./.grid_search.sh --env UNIFY_MODEL="gpt-4o|claude-3" --env UNIFY_CACHE="true|false" tests/

  # With additional pass-through options
  ./.grid_search.sh --env UNIFY_MODEL="gpt-4o|claude-3" --eval-only --wait tests/

  # Dry run to see what would be executed
  ./.grid_search.sh -n --env UNIFY_MODEL="gpt-4o|claude-3" tests/

  # Tag each run for easy filtering
  ./.grid_search.sh --env UNIFY_MODEL="gpt-4o|claude-3" --env UNIFY_TEST_TAGS="grid-exp-1" tests/

Notes:
  - Each combination spawns a separate .parallel_run.sh invocation
  - All runs execute concurrently by default (unless --wait-all is used)
  - Results are logged to the Combined context with the settings dict for filtering
  - Use --wait-all to run combinations sequentially (useful for resource-constrained environments)
USAGE
}

# Parse arguments
while (( "$#" )); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -n|--dry-run)
      DRY_RUN=1
      shift
      ;;
    --wait-all)
      WAIT_FOR_ALL=1
      shift
      ;;
    -e|--env)
      if [[ -n "${2:-}" ]]; then
        kv="$2"
        if [[ "$kv" == *"|"* ]]; then
          # Contains pipe - this is a grid variable
          GRID_VARS+=( "$kv" )
        else
          # No pipe - pass through as-is
          PASSTHROUGH+=( "--env" "$kv" )
        fi
        shift 2
      else
        echo "Error: --env requires KEY=VALUE argument" >&2
        exit 2
      fi
      ;;
    -*)
      # Other flags - pass through
      PASSTHROUGH+=( "$1" )
      shift
      ;;
    *)
      # Positional args (targets)
      TARGETS+=( "$1" )
      shift
      ;;
  esac
done

# If no grid variables, just run parallel_run.sh directly
if (( ${#GRID_VARS[@]} == 0 )); then
  echo "No grid variables specified (use --env KEY=val1|val2). Running single invocation..."
  exec "$PARALLEL_RUN" "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}" "${TARGETS[@]+"${TARGETS[@]}"}"
fi

# Parse grid variables into arrays
# GRID_KEYS[i] = key name
# GRID_VALUES[i] = "val1 val2 val3" (space-separated)
declare -a GRID_KEYS=()
declare -a GRID_VALUES=()

for entry in "${GRID_VARS[@]}"; do
  key="${entry%%=*}"
  values_str="${entry#*=}"
  # Replace | with space for iteration
  values_space="${values_str//|/ }"
  GRID_KEYS+=( "$key" )
  GRID_VALUES+=( "$values_space" )
done

# Generate all combinations using recursive expansion
# We'll build an array of "KEY1=val KEY2=val ..." strings
generate_combinations() {
  local depth="$1"
  local prefix="$2"

  if (( depth >= ${#GRID_KEYS[@]} )); then
    # Base case: output the combination
    echo "$prefix"
    return
  fi

  local key="${GRID_KEYS[$depth]}"
  local values="${GRID_VALUES[$depth]}"

  for val in $values; do
    local new_prefix
    if [[ -z "$prefix" ]]; then
      new_prefix="$key=$val"
    else
      new_prefix="$prefix $key=$val"
    fi
    generate_combinations $((depth + 1)) "$new_prefix"
  done
}

# Collect all combinations
mapfile -t COMBINATIONS < <(generate_combinations 0 "")

echo "Grid Search Configuration"
echo "========================="
echo "Grid variables:"
for i in "${!GRID_KEYS[@]}"; do
  echo "  ${GRID_KEYS[$i]}: ${GRID_VALUES[$i]// / | }"
done
echo ""
echo "Total combinations: ${#COMBINATIONS[@]}"
echo ""

# Show what we'll run
echo "Generated runs:"
for i in "${!COMBINATIONS[@]}"; do
  combo="${COMBINATIONS[$i]}"
  echo "  [$((i+1))/${#COMBINATIONS[@]}] $combo"
done
echo ""

if (( DRY_RUN )); then
  echo "Dry run - commands that would be executed:"
  echo ""
  for combo in "${COMBINATIONS[@]}"; do
    # Build the command
    cmd=( "$PARALLEL_RUN" )
    for kv in $combo; do
      cmd+=( "--env" "$kv" )
    done
    cmd+=( "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}" )
    cmd+=( "${TARGETS[@]+"${TARGETS[@]}"}" )
    echo "  ${cmd[*]}"
  done
  exit 0
fi

# Execute all combinations
echo "Launching ${#COMBINATIONS[@]} parallel runs..."
echo ""

declare -a PIDS=()

for i in "${!COMBINATIONS[@]}"; do
  combo="${COMBINATIONS[$i]}"

  # Build the command
  cmd=( "$PARALLEL_RUN" )
  for kv in $combo; do
    cmd+=( "--env" "$kv" )
  done
  cmd+=( "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}" )
  cmd+=( "${TARGETS[@]+"${TARGETS[@]}"}" )

  # Add --wait if running sequentially
  if (( WAIT_FOR_ALL )); then
    cmd+=( "--wait" )
  fi

  echo "[$((i+1))/${#COMBINATIONS[@]}] Running: ${cmd[*]}"

  if (( WAIT_FOR_ALL )); then
    # Run sequentially with --wait
    "${cmd[@]}"
    status=$?
    if (( status != 0 )); then
      echo "Warning: Run [$((i+1))] exited with status $status"
    fi
  else
    # Run in background
    "${cmd[@]}" &
    PIDS+=( $! )
  fi
done

# If running concurrently, wait for all to complete
if (( ! WAIT_FOR_ALL && ${#PIDS[@]} > 0 )); then
  echo ""
  echo "All ${#PIDS[@]} runs launched. Sessions are running in tmux."
  echo ""
  echo "To monitor progress:"
  echo "  watch -n 0.5 'tmux ls'"
  echo ""
  echo "Results will be logged to the Combined context with settings for filtering."
fi

echo ""
echo "Grid search complete."
