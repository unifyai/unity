#!/bin/bash

# Function to handle shutdown
cleanup() {
    echo "Shutting down processes..."
    if [ ! -z "$pid1" ] && kill -0 $pid1 2>/dev/null; then
        kill -TERM $pid1 2>/dev/null
    fi
    if [ ! -z "$pid2" ] && kill -0 $pid2 2>/dev/null; then
        kill -TERM $pid2 2>/dev/null
    fi

    # Wait a bit for graceful shutdown
    sleep 2

    # Force kill if still running
    if [ ! -z "$pid1" ] && kill -0 $pid1 2>/dev/null; then
        kill -KILL $pid1 2>/dev/null
    fi
    if [ ! -z "$pid2" ] && kill -0 $pid2 2>/dev/null; then
        kill -KILL $pid2 2>/dev/null
    fi

    wait $pid1 $pid2 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start the first process with unbuffered output
echo "Starting event manager..."
python -u event_manager.py &
pid1=$!

sleep 3

echo "Starting comms manager..."
python -u comms_manager.py &
pid2=$!

echo "Both processes started successfully!"
echo "Event manager PID: $pid1"
echo "Comms manager PID: $pid2"
echo "Press Ctrl+C to stop all processes..."

# Keep the script running and wait for processes
wait
