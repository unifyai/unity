#!/usr/bin/env bash
set -euo pipefail

# Optionally source environment from ../.env (relative to tests directory)
# Useful to provide UNIFY_KEY, UNIFY_BASE_URL, etc. from ~/unity/.env
if [ -f "../.env" ]; then
  # shellcheck disable=SC1091
  set -a
  . "../.env"
  set +a
fi

# Delete Unify projects whose names start with a given prefix (default: "UnityTests_").
# Useful after abrupt test termination (e.g., `tmux kill-server`) leaving temp projects behind.

API_BASE=""
PREFIX="UnityTests_"
ASSUME_YES=0
DRY_RUN=0
EXPLICIT_ENV=""
INCLUDE_MAIN=0

usage() {
  cat <<'USAGE'
Usage: .project_cleanup.sh [--dry-run] [-y|--yes] [--prefix PREFIX] [--include_main] [--staging|-s|--production|-p]

Options:
  --dry-run           Show matching projects without deleting
  -y, --yes           Do not prompt for confirmation
  --prefix PREFIX     Name prefix to match (default: UnityTests_)
  --include_main      Also delete the base UnityTests project
  -s, --staging       Use staging environment (skips prompt)
  -p, --production    Use production environment (skips prompt)
  -h, --help          Show this help

Environment:
  UNIFY_KEY           Required. API key for https://api.unify.ai
  UNIFY_BASE_URL      Optional. Full base URL including /v0; if set, skips env prompt
USAGE
}

while (( "$#" )); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    -y|--yes)
      ASSUME_YES=1
      ;;
    --prefix)
      shift
      if [[ $# -eq 0 || -z "${1:-}" ]]; then
        echo "Error: --prefix requires a value" >&2
        exit 2
      fi
      PREFIX="$1"
      ;;
    --include_main)
      INCLUDE_MAIN=1
      ;;
    -s|--staging)
      EXPLICIT_ENV="staging"
      ;;
    -p|--production)
      EXPLICIT_ENV="production"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift || true
done

if [[ -z "${UNIFY_KEY:-}" ]]; then
  echo "Error: UNIFY_KEY environment variable is not set." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required to parse API responses" >&2
  exit 1
fi

# Resolve API base URL
if [[ -n "${UNIFY_BASE_URL:-}" ]]; then
  API_BASE="$UNIFY_BASE_URL"
else
  if [[ -z "$EXPLICIT_ENV" ]]; then
    read -r -p "Select Unify environment: [s]taging or [p]roduction? (default: p) " env_ans
    env_ans_lc=$(printf '%s' "$env_ans" | tr '[:upper:]' '[:lower:]')
    case "$env_ans_lc" in
      s|staging) EXPLICIT_ENV="staging" ;;
      p|production|"") EXPLICIT_ENV="production" ;;
      *) EXPLICIT_ENV="production" ;;
    esac
  fi
  if [[ "$EXPLICIT_ENV" == "staging" ]]; then
    API_BASE="https://orchestra-staging-lz5fmz6i7q-ew.a.run.app/v0"
  else
    API_BASE="https://api.unify.ai/v0"
  fi
fi

echo "Listing projects from $API_BASE ..." >&2
resp="$(
  curl -sS -f \
    -H "Authorization: Bearer $UNIFY_KEY" \
    "$API_BASE/projects"
)" || {
  echo "Error: Failed to list projects" >&2
  exit 1
}

# Extract matching (id, name) pairs. Support both top-level array and {projects: [...]} shapes.
matches=()
tmp_matches="$(mktemp)"
trap 'rm -f "$tmp_matches"' EXIT
jq -r --arg pfx "$PREFIX" '
  (if type=="array" then . else (.projects // []) end)
  | map(
      if type=="string" then {id: ., name: .}
      else {id: (.id // .project_id // .projectId // .projectID // .uuid // .name // empty), name: (.name // .id // empty)}
      end
    )
  | .[]
  | select(.name? and (.name | type=="string") and (.name | startswith($pfx)))
  | select(.id != null and (.id | tostring) != "")
  | [.id, .name] | @tsv
' <<<"$resp" > "$tmp_matches" || true

# Optionally include the main UnityTests project (exact name match)
if (( INCLUDE_MAIN )); then
  jq -r '
    (if type=="array" then . else (.projects // []) end)
    | map(
        if type=="string" then {id: ., name: .}
        else {id: (.id // .project_id // .projectId // .projectID // .uuid // .name // empty), name: (.name // .id // empty)}
        end
      )
    | .[]
    | select(.name? and (.name | type=="string") and (.name == "UnityTests"))
    | select(.id != null and (.id | tostring) != "")
    | [.id, .name] | @tsv
  ' <<<"$resp" >> "$tmp_matches" || true
fi

while IFS= read -r line; do
  [[ -n "$line" ]] || continue
  matches+=( "$line" )
done < "$tmp_matches"

if (( ${#matches[@]} == 0 )); then
  echo "No projects found with prefix '$PREFIX'. Nothing to do." >&2
  exit 0
fi

echo "Found ${#matches[@]} project(s) to delete (prefix='$PREFIX'):" >&2
for m in "${matches[@]}"; do printf '  - %s\n' "$m"; done >&2

if (( DRY_RUN )); then
  echo "--dry-run specified; not deleting." >&2
  exit 0
fi

if (( ! ASSUME_YES )); then
  read -r -p "Proceed to delete ${#matches[@]} project(s)? [y/N] " ans
  ans_lc=$(printf '%s' "$ans" | tr '[:upper:]' '[:lower:]')
  case "$ans_lc" in
    y|yes) ;;
    *) echo "Aborted." >&2; exit 1 ;;
  esac
fi

CONCURRENCY="${CONCURRENCY:-8}"
tmp_results="$(mktemp)"
# Extend the existing trap to also clean up tmp_results
trap 'rm -f "$tmp_matches" "$tmp_results"' EXIT

# Fire deletions concurrently; one record per job, capture output for summary
printf '%s\0' "${matches[@]}" \
| UNIFY_KEY="$UNIFY_KEY" API_BASE="$API_BASE" xargs -0 -n1 -P "$CONCURRENCY" bash -c '
  IFS=$'"'\t'"' read -r proj_id proj_name <<<"$1"
  if curl -sS -f -X DELETE \
       -H "Authorization: Bearer $UNIFY_KEY" \
       "$API_BASE/project/$proj_id" >/dev/null; then
    printf "Deleted: %s (%s)\n" "$proj_id" "$proj_name"
  else
    printf "Failed:  %s (%s)\n" "$proj_id" "$proj_name" >&2
    exit 1
  fi
' _ \
2>&1 | tee -a "$tmp_results"

deleted=$(grep -c '^Deleted:' "$tmp_results" || true)
failed=$(grep -c '^Failed:' "$tmp_results" || true)
echo "Done. Deleted=$deleted Failed=$failed"
