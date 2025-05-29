#!/bin/bash

# Function to handle shutdown
cleanup() {
    echo "Shutting down processes..."
    if [ ! -z "$pid1" ] && kill -0 $pid1 2>/dev/null; then
        kill -TERM $pid1 2>/dev/null
    fi

    # Wait a bit for graceful shutdown
    sleep 2

    # Force kill if still running
    if [ ! -z "$pid1" ] && kill -0 $pid1 2>/dev/null; then
        kill -KILL $pid1 2>/dev/null
    fi

    wait $pid1 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start the first process with unbuffered output
echo "Starting event manager..."
python -u main.py &
pid1=$!

# Keep the script running and wait for processes
wait
