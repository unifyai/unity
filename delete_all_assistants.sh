#!/usr/bin/env bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_ENV_FILE="$_SCRIPT_DIR/.env"
if [ -f "$_ENV_FILE" ]; then
  set -a
  . "$_ENV_FILE"
  set +a
fi
unset _ENV_FILE _SCRIPT_DIR

if [[ -z "${UNIFY_KEY:-}" ]]; then
  echo "Error: UNIFY_KEY not set (check .env)" >&2
  exit 1
fi

API_BASE="https://api.staging.internal.saas.unify.ai/v0"

echo "Listing assistants from $API_BASE ..."
list_resp="$(curl -sS -f -H "Authorization: Bearer $UNIFY_KEY" "$API_BASE/assistant")" || {
  echo "Error: Failed to list assistants" >&2
  exit 1
}

count="$(jq '.info | length' <<<"$list_resp")"
if (( count == 0 )); then
  echo "No assistants found."
else
  echo "Found $count assistant(s). Deleting ..."

  jq -c '.info[]' <<<"$list_resp" | while IFS= read -r assistant; do
    agent_id="$(jq -r '.agent_id' <<<"$assistant")"
    first="$(jq -r '.first_name // ""' <<<"$assistant")"
    last="$(jq -r '.surname // ""' <<<"$assistant")"
    name="${first}${last:+ $last}"

    curl -sS -f -X DELETE \
      -H "Authorization: Bearer $UNIFY_KEY" \
      "$API_BASE/assistant/$agent_id" >/dev/null || {
      echo "FAILED to delete $name (id=$agent_id)" >&2
      continue
    }

    verify_resp="$(curl -sS -f -H "Authorization: Bearer $UNIFY_KEY" "$API_BASE/assistant")" || {
      echo "FAILED to verify deletion of $name (id=$agent_id)" >&2
      continue
    }
    still_exists="$(jq --arg id "$agent_id" '[.info[] | select(.agent_id == $id)] | length' <<<"$verify_resp")"
    if (( still_exists > 0 )); then
      echo "WARNING: $name (id=$agent_id) still exists after deletion" >&2
    else
      echo "$name deleted"
    fi
  done
fi

echo "Deleting Assistants project ..."
del_resp="$(curl -sS -w '\n%{http_code}' -X DELETE \
  -H "Authorization: Bearer $UNIFY_KEY" \
  "$API_BASE/project/Assistants")"
http_code="$(tail -1 <<<"$del_resp")"
body="$(sed '$d' <<<"$del_resp")"

if [[ "$http_code" == "200" ]]; then
  echo "Assistants project deleted"
elif [[ "$http_code" == "404" ]]; then
  echo "Assistants project not found (already deleted or never existed)"
else
  echo "FAILED to delete Assistants project (HTTP $http_code): $body" >&2
  exit 1
fi
