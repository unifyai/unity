#!/usr/bin/env bash

set -euo pipefail

SESSION_NAME=${SESSION_NAME:-intranet}
API_WINDOW=${API_WINDOW:-api}
COMMS_WINDOW=${COMMS_WINDOW:-comms}
API_PORT=${API_PORT:-8000}
KILL_PORT=${KILL_PORT:-0}
CLEAR_LOGS=${CLEAR_LOGS:-1}
OPEN_LOGS_WINDOW=${OPEN_LOGS_WINDOW:-0}
ATTACH_LOGS=${ATTACH_LOGS:-0}

API_LOG=${API_LOG:-/tmp/intranet_api.log}
COMMS_LOG=${COMMS_LOG:-/tmp/intranet_comms.log}

API_CMD='cd "$HOME"/unity && source .unity/bin/activate && python intranet/scripts/02_start_api.py --workers 4'
COMMS_CMD='cd "$HOME"/unity && source .unity/bin/activate && python intranet/core/comms.py'

echo "[intranet] Starting services in tmux (detached)"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[intranet] tmux is not installed. Install it with: sudo apt-get update && sudo apt-get install -y tmux"
  exit 1
fi

if [ ! -f "$HOME/unity/.unity/bin/activate" ]; then
  echo "[intranet] Warning: Python venv not found at $HOME/unity/.unity/bin/activate"
  echo "[intranet] The services may fail to start if the environment is missing."
fi

timestamp() { date -Is; }

# --- argument parsing (optional) ---
while [ $# -gt 0 ]; do
  case "$1" in
    --kill-port)
      KILL_PORT=1
      ;;
    --no-clear-logs)
      CLEAR_LOGS=0
      ;;
    --logs)
      OPEN_LOGS_WINDOW=1
      ;;
    --attach-logs)
      OPEN_LOGS_WINDOW=1
      ATTACH_LOGS=1
      ;;
    --api-port)
      shift
      API_PORT=${1:-8000}
      ;;
    --session)
      shift
      SESSION_NAME=${1:-intranet}
      ;;
    *)
      ;;
  esac
  shift || true
done

find_pids_on_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$port" 2>/dev/null | tr -d ' ' || true
  else
    ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p {print $6}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' || true
  fi
}

require_port_free() {
  local port="$1"
  local pids
  pids=$(find_pids_on_port "$port")
  if [ -n "$pids" ]; then
    echo "[intranet] Port $port is in use by PID(s): $pids"
    if [ "$KILL_PORT" = "1" ]; then
      echo "[intranet] Attempting to stop process(es) on port $port..."
      for pid in $pids; do
        if kill -TERM "$pid" 2>/dev/null; then
          echo "[intranet] Sent SIGTERM to $pid"
        fi
      done
      # wait up to 5s for port to free
      for _ in 1 2 3 4 5; do
        sleep 1
        pids=$(find_pids_on_port "$port")
        [ -z "$pids" ] && break
      done
      if [ -n "$pids" ]; then
        echo "[intranet] Forcing stop of remaining PID(s): $pids"
        for pid in $pids; do
          kill -KILL "$pid" 2>/dev/null || true
        done
      fi
      pids=$(find_pids_on_port "$port")
      if [ -n "$pids" ]; then
        echo "[intranet] Port $port is still in use. Aborting."
        exit 1
      else
        echo "[intranet] Port $port freed successfully."
      fi
    else
      echo "[intranet] Aborting start. To auto-kill the process on port $port, re-run with --kill-port"
      echo "[intranet] Or manually inspect with:"
      echo "           ss -ltnp | grep :$port || lsof -iTCP:$port -sTCP:LISTEN"
      exit 1
    fi
  fi
}

start_api_window() {
  # Ensure window exists fresh, then start command with logging
  if tmux list-windows -t "$SESSION_NAME" -F '#W' 2>/dev/null | grep -Fxq "$API_WINDOW"; then
    tmux kill-window -t "$SESSION_NAME:$API_WINDOW" || true
  fi
  tmux new-window -t "$SESSION_NAME" -n "$API_WINDOW"
  echo "==== $(timestamp) Starting API service ====\n" >> "$API_LOG"
  tmux send-keys -t "$SESSION_NAME:$API_WINDOW" "${API_CMD} 2>&1 | tee -a '$API_LOG'" C-m
}

start_comms_window() {
  # Ensure window exists fresh, then start command with logging
  if tmux list-windows -t "$SESSION_NAME" -F '#W' 2>/dev/null | grep -Fxq "$COMMS_WINDOW"; then
    tmux kill-window -t "$SESSION_NAME:$COMMS_WINDOW" || true
  fi
  tmux new-window -t "$SESSION_NAME" -n "$COMMS_WINDOW"
  echo "==== $(timestamp) Starting COMMS service ====\n" >> "$COMMS_LOG"
  tmux send-keys -t "$SESSION_NAME:$COMMS_WINDOW" "${COMMS_CMD} 2>&1 | tee -a '$COMMS_LOG'" C-m
}

require_port_free "$API_PORT"

# Prepare log files
mkdir -p "$(dirname "$API_LOG")" "$(dirname "$COMMS_LOG")"
if [ "$CLEAR_LOGS" = "1" ]; then
  : > "$API_LOG"
  : > "$COMMS_LOG"
  echo "[intranet] Cleared logs: $API_LOG and $COMMS_LOG"
else
  echo "[intranet] Preserving existing logs (use --no-clear-logs to toggle)"
fi

create_logs_window() {
  local LOGS_WINDOW="logs"
  if tmux list-windows -t "$SESSION_NAME" -F '#W' 2>/dev/null | grep -Fxq "$LOGS_WINDOW"; then
    tmux kill-window -t "$SESSION_NAME:$LOGS_WINDOW" || true
  fi
  tmux new-window -t "$SESSION_NAME" -n "$LOGS_WINDOW"
  tmux send-keys -t "$SESSION_NAME:$LOGS_WINDOW" "tail -f '$API_LOG'" C-m
  tmux split-window -h -t "$SESSION_NAME:$LOGS_WINDOW"
  tmux send-keys -t "$SESSION_NAME:$LOGS_WINDOW" "tail -f '$COMMS_LOG'" C-m
  tmux select-layout -t "$SESSION_NAME:$LOGS_WINDOW" even-horizontal
}

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[intranet] tmux session '$SESSION_NAME' already exists. Restarting windows..."
  start_api_window
  start_comms_window
  if [ "$OPEN_LOGS_WINDOW" = "1" ]; then
    create_logs_window
  fi
else
  echo "[intranet] Creating tmux session '$SESSION_NAME'"
  tmux new-session -d -s "$SESSION_NAME" -n "$API_WINDOW"
  echo "==== $(timestamp) Starting API service ====\n" >> "$API_LOG"
  tmux send-keys -t "$SESSION_NAME:$API_WINDOW" "${API_CMD} 2>&1 | tee -a '$API_LOG'" C-m
  start_comms_window
  if [ "$OPEN_LOGS_WINDOW" = "1" ]; then
    create_logs_window
  fi
fi

echo "[intranet] Services started in detached tmux session: $SESSION_NAME"
echo
echo "Manage & inspect:"
echo "  - Attach to session:         tmux attach -t $SESSION_NAME"
echo "  - Switch to API window:      tmux select-window -t $SESSION_NAME:$API_WINDOW"
echo "  - Switch to COMMS window:    tmux select-window -t $SESSION_NAME:$COMMS_WINDOW"
echo
echo "Tail logs without attaching:"
echo "  - API logs:   tail -f $API_LOG"
echo "  - COMMS logs: tail -f $COMMS_LOG"
echo "  - Side-by-side in tmux:  tmux attach -t $SESSION_NAME; then: tmux select-window -t $SESSION_NAME:logs"
echo "    (Or start with logs pane via: intranet/scripts/start_intranet_service.sh --logs)"
echo
echo "List/kill sessions:"
echo "  - List sessions:              tmux ls"
echo "  - Kill the session:           tmux kill-session -t $SESSION_NAME"
echo
echo "Restart a single service:"
echo "  - Restart API:   tmux kill-window -t $SESSION_NAME:$API_WINDOW && tmux new-window -t $SESSION_NAME -n $API_WINDOW && tmux send-keys -t $SESSION_NAME:$API_WINDOW \"${API_CMD} 2>&1 | tee -a '$API_LOG'\" C-m"
echo "  - Restart COMMS: tmux kill-window -t $SESSION_NAME:$COMMS_WINDOW && tmux new-window -t $SESSION_NAME -n $COMMS_WINDOW && tmux send-keys -t $SESSION_NAME:$COMMS_WINDOW \"${COMMS_CMD} 2>&1 | tee -a '$COMMS_LOG'\" C-m"
echo
echo "[intranet] Done."

# Optional: attach directly to logs window
if [ "$ATTACH_LOGS" = "1" ]; then
  echo "[intranet] Attaching to logs window (Ctrl+b d to detach)"
  tmux select-window -t "$SESSION_NAME:logs" 2>/dev/null || true
  tmux attach -t "$SESSION_NAME"
fi
