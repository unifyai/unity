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
import signal
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Query
import uvicorn
import unity.service

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


# Endpoint 1: Start the service
@app.post("/start")
async def start_service(auth: bool = Depends(require_auth)):
    """Start the Unity service"""
    global service_status

    if unity.service.start():
        service_status = "running"
        return {
            "status": "started",
            "message": "Unity service started successfully",
            "pid": (
                unity.service.get_process().pid if unity.service.get_process() else None
            ),
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

    unity.service.stop("manual_stop")
    service_status = "stopped"

    return {"status": "stopped", "message": "Unity service stopped manually"}


# Endpoint 3: Get status
@app.get("/status")
async def get_status(auth: bool = Depends(require_auth)):
    """Get current status of the Unity service"""
    return unity.service.get_status()


# Health check for the wrapper service
@app.get("/health")
async def health_check(auth: bool = Depends(require_auth)):
    return {
        "wrapper_status": "healthy",
        "unity_service": unity.service.get_status(),
        "assistant_id": os.environ.get("ASSISTANT_ID", "default"),
    }


# Graceful shutdown handler
def signal_handler(signum, frame):
    print("Shutting down Unity service...")
    unity.service.stop("signal_shutdown")
    exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Auto-start Unity service if configured
auto_start = os.environ.get("AUTO_START_UNITY", "false").lower() == "true"
if auto_start:
    print("Auto-starting Unity service...")
    unity.service.start()

if __name__ == "__main__":
    # Only run in development mode if called directly
    print("Running in development mode...")
    uvicorn.run(
        "wrapper_app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )
