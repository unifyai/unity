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
import signal
import threading
import subprocess
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Query
import uvicorn
from new_terminal_helper import terminate_process

app = FastAPI()

# Global state for the single Unity service
unity_process = None
service_status = "stopped"


# Authentication dependency
async def require_auth(
    authorization: Optional[str] = Header(None),
    admin_key: Optional[str] = Query(None),
):
    """Authentication dependency for FastAPI"""
    auth_admin_key = None

    # Extract Bearer token from Authorization header
    if authorization and authorization.startswith("Bearer "):
        auth_admin_key = authorization[7:]  # Remove 'Bearer ' prefix
    else:
        # Fallback to query parameter
        auth_admin_key = admin_key

    if not auth_admin_key:
        raise HTTPException(status_code=401, detail="Admin key required")

    # First check if the provided key matches the admin key from environment
    if auth_admin_key == os.environ.get("ORCHESTRA_ADMIN_KEY"):
        return True

    # If neither condition is met, raise unauthorized exception
    raise HTTPException(status_code=401, detail="Unauthorized")


class UnityServiceManager:
    def __init__(self):
        self.process = None
        self.start_time = None
        self.shutdown_reason = None  # Track why the service stopped
        self.monitor_thread = None
        self.monitoring = False

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

            self.process = subprocess.Popen(
                [sys.executable, "main.py"],
                start_new_session=True,
            )

            self.start_time = time.time()

            # Give it a moment to start
            time.sleep(2)

            # Check if process is still running (didn't crash immediately)
            if self.process.poll() is None:
                print("Unity service started successfully")
                self.shutdown_reason = None  # Clear any previous shutdown reason
                self._start_monitoring()
                return True
            else:
                print("Unity service failed to start (process exited)")
                self.shutdown_reason = "startup_failure"
                return False

        except Exception as e:
            print(f"Failed to start Unity service: {e}")
            return False

    def stop_unity_service(self, reason="manual_stop"):
        """Stop the main.py subprocess and all its children (like call.py)"""
        self._stop_monitoring()  # Stop monitoring first

        if self.process:
            try:
                print("Stopping Unity service and all child processes...")
                # Use the terminate_process function which handles process groups properly
                terminate_process(self.process)
                print("Unity service and child processes stopped")
                self.shutdown_reason = reason
            except Exception as e:
                print(f"Error stopping Unity service: {e}")
                self.shutdown_reason = f"stop_error: {e}"

            self.process = None
            return True
        return True

    def _start_monitoring(self):
        """Start background monitoring of the Unity process"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_process,
                daemon=True,
            )
            self.monitor_thread.start()

    def _stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring = False

    def _monitor_process(self):
        """Background thread to monitor process health"""
        while self.monitoring and self.process:
            try:
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process has exited
                    exit_code = self.process.poll()
                    if exit_code == 0 and not self.shutdown_reason:
                        # Clean exit without explicit reason - likely inactivity timeout
                        self.shutdown_reason = "inactivity_timeout"
                        print(
                            "Unity service exited cleanly - likely due to inactivity timeout",
                        )
                    elif exit_code != 0 and not self.shutdown_reason:
                        self.shutdown_reason = (
                            f"process_crashed (exit_code: {exit_code})"
                        )
                        print(f"Unity service crashed with exit code: {exit_code}")

                    self.monitoring = False
                    break

                # Check every 10 seconds
                time.sleep(10)

            except Exception as e:
                print(f"Error in process monitoring: {e}")
                self.monitoring = False
                break

    def is_running(self):
        """Check if the main.py subprocess is running"""
        return self.process and self.process.poll() is None

    def get_status(self):
        """Get detailed status of the Unity service instance"""
        running = self.is_running()
        uptime = time.time() - self.start_time if self.start_time and running else 0

        status = {
            "running": running,
            "uptime_seconds": uptime,
            "process_id": self.process.pid if self.process else None,
            "assistant_id": os.environ.get("ASSISTANT_ID", "default"),
            "shutdown_reason": self.shutdown_reason,
            "inactivity_timeout_minutes": 6,  # Document the timeout setting
        }

        # Add additional context based on shutdown reason
        if self.shutdown_reason == "inactivity_timeout":
            status["message"] = "Service shut down due to 6 minutes of inactivity"
        elif self.shutdown_reason == "manual_stop":
            status["message"] = "Service stopped manually via API"
        elif self.shutdown_reason and "process_crashed" in self.shutdown_reason:
            status["message"] = "Service process crashed unexpectedly"

        return status


# Initialize the service manager
unity_manager = UnityServiceManager()


# Endpoint 1: Start the service
@app.post("/start")
async def start_service(auth: bool = Depends(require_auth)):
    """Start the Unity service"""
    global service_status

    if unity_manager.start_unity_service():
        service_status = "running"
        return {
            "status": "started",
            "message": "Unity service started successfully",
            "pid": unity_manager.process.pid if unity_manager.process else None,
            "assistant_id": os.environ.get("ASSISTANT_ID", "default"),
        }
    else:
        service_status = "failed"
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "Failed to start Unity service"},
        )


# Endpoint 2: Stop the service
@app.post("/stop")
async def stop_service(auth: bool = Depends(require_auth)):
    """Stop the Unity service"""
    global service_status

    unity_manager.stop_unity_service("manual_stop")
    service_status = "stopped"

    return {"status": "stopped", "message": "Unity service stopped manually"}


# Endpoint 3: Get status
@app.get("/status")
async def get_status(auth: bool = Depends(require_auth)):
    """Get current status of the Unity service"""
    return unity_manager.get_status()


# Health check for the wrapper service
@app.get("/health")
async def health_check(auth: bool = Depends(require_auth)):
    return {
        "wrapper_status": "healthy",
        "unity_service": unity_manager.get_status(),
        "assistant_id": os.environ.get("ASSISTANT_ID", "default"),
    }


# Graceful shutdown handler
def signal_handler(signum, frame):
    print("Shutting down Unity service...")
    unity_manager.stop_unity_service("signal_shutdown")
    exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Auto-start Unity service if configured
auto_start = os.environ.get("AUTO_START_UNITY", "false").lower() == "true"
if auto_start:
    print("Auto-starting Unity service...")
    unity_manager.start_unity_service()

if __name__ == "__main__":
    # Only run in development mode if called directly
    print("Running in development mode...")
    uvicorn.run(
        "wrapper_app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )
