#!/usr/bin/env bash
# Delete all action events from the local Orchestra database.
# Keeps the project, contexts, assistants, and login credentials intact.
#
# Usage:
#   scripts/dev/clear_action_events.sh

set -euo pipefail

CONTAINER=$(docker ps --filter "publish=5432" --format "{{.Names}}" | head -1)

if [[ -z "$CONTAINER" ]]; then
  echo "No PostgreSQL container found on port 5432" >&2
  exit 1
fi

DELETED=$(docker exec "$CONTAINER" psql -U orchestra -d orchestra -tAc \
  "DELETE FROM log_event WHERE project_id = (SELECT id FROM project WHERE name = 'Assistants') RETURNING id;" \
  | wc -l | tr -d ' ')

echo "Deleted $DELETED event(s)"
