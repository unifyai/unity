# Starts the Unity service
import signal
import unity.service

unity.service.start()
service_status = "stopped"

# Graceful shutdown handler
def signal_handler(signum, frame):
    print("Shutting down Unity service...")
    unity.service.stop("signal_shutdown")
    # Don't exit the container - let uvicorn handle the shutdown
    exit(0)  # Removed this line to prevent container exit


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("Starting Unity service...")
    unity.service.start()
