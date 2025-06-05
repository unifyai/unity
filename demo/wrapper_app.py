# Cloud Run wrapper service that controls a single Unity assistant service
#
# ARCHITECTURE:
# This wrapper app manages ONE instance of the Unity event manager (main.py).
# Each assistant gets their own Cloud Run deployment with their own wrapper_app.py
#
# How it works:
# 1. This wrapper starts/stops a single main.py subprocess for this assistant
# 2. The main.py instance runs the Unity event manager with its own listeners
# 3. This wrapper provides REST endpoints to start/stop/manage the main.py instance
# 4. Messages are handled directly by main.py via its own pubsub/listeners
#
# Endpoints:
# - POST /start     - Start the main.py instance
# - POST /stop      - Stop the main.py instance
# - GET  /status    - Get status of the instance
# - GET  /health    - Health check

import os
import time
import subprocess
import signal
from flask import Flask, jsonify
from new_terminal_helper import run_script, terminate_process

app = Flask(__name__)

# Global state for the single Unity service
unity_process = None
service_status = "stopped"


class UnityServiceManager:
    def __init__(self):
        self.process = None
        self.start_time = None

    def start_unity_service(self):
        """Start main.py (the Unity event manager) as a subprocess with process group"""
        if self.process and self.process.poll() is None:
            print("Unity service is already running")
            return True  # Already running

        try:
            # Get environment variables for this assistant (set by Cloud Run)
            assistant_id = os.environ.get("ASSISTANT_ID", "default")

            # Start main.py using run_script which handles process groups automatically
            print(f"Starting Unity service (main.py) for assistant {assistant_id}")
            self.process = run_script("main.py")
            self.start_time = time.time()

            # Give it a moment to start
            time.sleep(2)

            # Check if process is still running (didn't crash immediately)
            if self.process.poll() is None:
                print("Unity service started successfully")
                return True
            else:
                print("Unity service failed to start (process exited)")
                return False

        except Exception as e:
            print(f"Failed to start Unity service: {e}")
            return False

    def stop_unity_service(self):
        """Stop the main.py subprocess and all its children (like call.py)"""
        if self.process:
            try:
                print("Stopping Unity service and all child processes...")
                # Use the terminate_process function which handles process groups properly
                terminate_process(self.process)
                print("Unity service and child processes stopped")
            except Exception as e:
                print(f"Error stopping Unity service: {e}")

            self.process = None
            return True
        return True

    def is_running(self):
        """Check if the main.py subprocess is running"""
        return self.process and self.process.poll() is None

    def get_status(self):
        """Get detailed status of the Unity service instance"""
        running = self.is_running()
        uptime = time.time() - self.start_time if self.start_time and running else 0

        return {
            "running": running,
            "uptime_seconds": uptime,
            "process_id": self.process.pid if self.process else None,
            "assistant_id": os.environ.get("ASSISTANT_ID", "default"),
        }


# Initialize the service manager
unity_manager = UnityServiceManager()

# Endpoint 1: Start the service
@app.route("/start", methods=["POST"])
def start_service():
    """Start the Unity service"""
    global service_status

    if unity_manager.start_unity_service():
        service_status = "running"
        return jsonify(
            {
                "status": "started",
                "message": "Unity service started successfully",
                "pid": unity_manager.process.pid if unity_manager.process else None,
                "assistant_id": os.environ.get("ASSISTANT_ID", "default"),
            }
        )
    else:
        service_status = "failed"
        return (
            jsonify({"status": "error", "message": "Failed to start Unity service"}),
            500,
        )


# Endpoint 2: Stop the service
@app.route("/stop", methods=["POST"])
def stop_service():
    """Stop the Unity service"""
    global service_status

    unity_manager.stop_unity_service()
    service_status = "stopped"

    return jsonify({"status": "stopped", "message": "Unity service stopped"})


# Endpoint 3: Get status
@app.route("/status", methods=["GET"])
def get_status():
    """Get current status of the Unity service"""
    return jsonify(unity_manager.get_status())


# Health check for the wrapper service
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "wrapper_status": "healthy",
            "unity_service": unity_manager.get_status(),
            "assistant_id": os.environ.get("ASSISTANT_ID", "default"),
        }
    )


# Graceful shutdown handler
def signal_handler(signum, frame):
    print("Shutting down Unity service...")
    unity_manager.stop_unity_service()
    exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    # Optionally auto-start the Unity service when wrapper starts
    auto_start = os.environ.get("AUTO_START_UNITY", "false").lower() == "true"
    if auto_start:
        print("Auto-starting Unity service...")
        unity_manager.start_unity_service()

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
