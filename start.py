# Starts the Unity service
import signal
import unity.service


# Graceful shutdown handler
def signal_handler(signum, frame):
    print("Shutting down Unity service...")
    unity.service.stop("signal_shutdown")
    exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("Starting Unity service...")
    unity.service.start()
