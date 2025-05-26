#!/bin/bash

# Function to prefix output lines
prefix_output() {
    local prefix="$1"
    while IFS= read -r line; do
        echo "[$prefix] $line"
    done
}

# Start event_manager.py with prefixed output
echo "Starting event_manager.py..."
python event_manager.py 2>&1 | prefix_output "EVENT_MGR" &
EVENT_MANAGER_PID=$!

# Give event manager time to start up
echo "Waiting for event manager to start..."
sleep 5

# Start comms_manager.py with prefixed output
echo "Starting comms_manager.py..."
python comms_manager.py 2>&1 | prefix_output "COMMS_MGR" &
COMMS_MANAGER_PID=$!

# Function to handle shutdown
cleanup() {
    echo "Shutting down processes..."
    kill $EVENT_MANAGER_PID $COMMS_MANAGER_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

echo "Both processes started successfully!"
echo "Event manager PID: $EVENT_MANAGER_PID"
echo "Comms manager PID: $COMMS_MANAGER_PID"

# Wait for both processes to finish
wait $EVENT_MANAGER_PID $COMMS_MANAGER_PID
