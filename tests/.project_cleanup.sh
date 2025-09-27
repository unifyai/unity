#!/usr/bin/env bash
set -euo pipefail

# Delete Unify projects whose names start with a given prefix (default: "UnityTests_").
# Useful after abrupt test termination (e.g., `tmux kill-server`) leaving temp projects behind.

UNIFY_API_BASE="${UNIFY_API:-https://api.unify.ai/v0}"
PREFIX="UnityTests_"
ASSUME_YES=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: .project_cleanup.sh [--dry-run] [-y|--yes] [--prefix PREFIX]

Options:
  --dry-run           Show matching projects without deleting
  -y, --yes           Do not prompt for confirmation
  --prefix PREFIX     Name prefix to match (default: UnityTests_)
  -h, --help          Show this help

Environment:
  UNIFY_KEY           Required. API key for https://api.unify.ai
  UNIFY_API           Optional. Override API base (default: https://api.unify.ai/v0)
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

echo "Listing projects from $UNIFY_API_BASE ..." >&2
resp="$(
  curl -sS -f \
    -H "Authorization: Bearer $UNIFY_KEY" \
    "$UNIFY_API_BASE/projects"
)" || {
  echo "Error: Failed to list projects" >&2
  exit 1
}

# Extract matching (id, name) pairs. Support both top-level array and {projects: [...]} shapes.
mapfile -t matches < <(
  jq -r --arg pfx "$PREFIX" '
    (if type=="array" then . else (.projects // []) end)
    | .[]
    | select(.name? and (.name | type=="string") and (.name | startswith($pfx)))
    | {id: (.id // .project_id // .projectId // .projectID // .uuid // empty), name}
    | select(.id != null and .id != "")
    | [.id, .name] | @tsv
  ' <<<"$resp"
)

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
  case "${ans,,}" in
    y|yes) ;;
    *) echo "Aborted." >&2; exit 1 ;;
  esac
fi

deleted=0
failed=0
for m in "${matches[@]}"; do
  IFS=$'\t' read -r proj_id proj_name <<<"$m"
  if curl -sS -f -X DELETE \
       -H "Authorization: Bearer $UNIFY_KEY" \
       "$UNIFY_API_BASE/project/$proj_id" >/dev/null; then
    echo "Deleted: $proj_id ($proj_name)"
    ((deleted++))
  else
    echo "Failed:  $proj_id ($proj_name)" >&2
    ((failed++))
  fi
  sleep 0.05
done

echo "Done. Deleted=$deleted Failed=$failed"
