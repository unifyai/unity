#!/bin/bash
set -euo pipefail

PASSFILE="/root/.vnc/passwd"
TMPFILE="${PASSFILE}.new"

if [ -z "${UNIFY_KEY:-}" ]; then
  echo "[update_vnc_password] UNIFY_KEY not set" >&2
  exit 1
fi

mkdir -p "$(dirname "$PASSFILE")"

# Write new password atomically
x11vnc -storepasswd "${UNIFY_KEY}" "${TMPFILE}"
chmod 600 "${TMPFILE}"
mv -f "${TMPFILE}" "${PASSFILE}"
chmod 600 "${PASSFILE}"

# Restart x11vnc to pick up new password
pkill x11vnc || true
nohup x11vnc -display :99 -rfbauth "${PASSFILE}" -forever -shared -bg -rfbport 5900 -rfbportv6 0 -noxdamage -nowf -nocursorshape -cursor arrow -nodpms >/dev/null 2>&1 &

echo "[update_vnc_password] VNC password updated and x11vnc restarted"
