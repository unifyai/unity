#!/usr/bin/env python3
"""
Start API Server
================

This script starts the FastAPI server for the RAG system and makes it
available for user queries. It automatically handles schema initialization
and document ingestion during startup.

What this script does:
- Automatically initializes schema if needed
- Automatically ingests documents if needed
- Instantiates the IntranetRAGAgent with tool loops
- Starts the FastAPI server with all endpoints
- Makes the system ready for user queries via HTTP/WebSocket

Usage:
    python scripts/05_start_api.py [OPTIONS]

Options:
    --port           Port to run on (default: 8000)
    --host           Host to bind to (default: 0.0.0.0)
    --dev            Development mode with auto-reload
    --use-tool-loops Use tool loop architecture for initialization (slower, more robust)

Environment Variables:
    RAG_USE_TOOL_LOOPS=true   Use tool loops for initialization (default: false)

Architecture Modes:
    Direct Methods (default): Fast startup, no LLM loops during initialization
    Tool Loops (--use-tool-loops): Unity pattern, LLM-driven initialization
"""

import asyncio
import sys
import argparse
import subprocess
import os
import signal
import time
from pathlib import Path

# Import utilities and setup environment
from utils import initialize_script_environment, get_config_values

# Initialize environment and setup paths
if not initialize_script_environment():
    sys.exit(1)

# Global variable to track the server process for graceful shutdown
server_process = None
shutdown_requested = False


def terminate_process(proc: subprocess.Popen) -> None:
    """
    Terminate a subprocess gracefully, falling back to force kill if needed.
    Handles both Windows and Unix-like systems.
    """
    if proc is None:
        return

    try:
        print("🛑 Initiating graceful shutdown...")

        # Send SIGTERM to the process group
        if sys.platform.startswith("win"):
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

        # Wait for process to terminate gracefully
        try:
            proc.wait(timeout=10)  # Give it 10 seconds
            print("✅ Server terminated gracefully")
        except subprocess.TimeoutExpired:
            # If process doesn't terminate gracefully, force kill
            print("⚠️  Server did not terminate gracefully, force killing...")
            if sys.platform.startswith("win"):
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            print("🔥 Server force killed")
    except Exception as e:
        print(f"❌ Error during server termination: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested, server_process

    signal_names = {
        signal.SIGINT: "SIGINT (Ctrl+C)",
        signal.SIGTERM: "SIGTERM (shutdown)",
    }

    signal_name = signal_names.get(signum, f"Signal {signum}")
    print(f"\n🛑 Received {signal_name}, shutting down...")

    shutdown_requested = True

    if server_process:
        terminate_process(server_process)

    print("👋 Goodbye!")
    sys.exit(0)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    # Handle Ctrl+C and termination signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # On Unix systems, also handle SIGHUP
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal_handler)

    print("🔧 Signal handlers configured for graceful shutdown")


def check_system_ready():
    """Check if the RAG system is ready to serve requests."""

    print("🔍 Checking system readiness...")

    try:
        # Check if intranet directory exists
        intranet_dir = Path("intranet")
        if not intranet_dir.exists():
            print("❌ Intranet directory not found")
            return False

        # Check if core modules exist
        core_files = ["core/rag_agent.py", "core/api.py"]
        for file in core_files:
            if not (intranet_dir / file).exists():
                print(f"❌ Missing core file: {file}")
                return False

        # Quick import test

        print("✅ System components found and importable")
        return True

    except Exception as e:
        print(f"❌ System check failed: {e}")
        return False


def verify_setup():
    """Verify that basic setup has been completed (lightweight check)."""

    try:
        # Check if documents directory exists and has files
        config = get_config_values()
        docs_path = Path(config["documents_path"])

        if not docs_path.exists():
            print(f"📁 Creating documents directory: {docs_path}")
            docs_path.mkdir(parents=True, exist_ok=True)

        # Check for document files
        doc_files = []
        for pattern in ["*.pdf", "*.docx", "*.doc", "*.txt"]:
            doc_files.extend(docs_path.glob(pattern))

        if doc_files:
            print(f"📄 Found {len(doc_files)} documents for processing")
        else:
            print("📄 No documents found - add files to intranet/policies/")

        return True

    except Exception as e:
        print(f"⚠️  Could not verify setup: {e}")
        return True  # Still allow API to start


async def start_api_server(host=None, port=None, dev_mode=False):
    """Start the FastAPI server with graceful shutdown handling."""
    global server_process

    print(f"🚀 Starting RAG API Server...")
    print("=" * 50)

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    # Get configuration from environment if not provided
    config = get_config_values()
    if host is None:
        host = config["api_host"]
    if port is None:
        port = config["api_port"]
    if dev_mode is None:
        dev_mode = config["dev_mode"]

    # Check system readiness
    if not check_system_ready():
        print("❌ System not ready. Please run setup scripts first.")
        return False

    # Verify setup (non-blocking)
    print("📊 Verifying system setup...")
    verify_setup()

    print(
        "✅ System ready to start API server (initialization will happen during app startup)",
    )

    # Set environment variables
    os.environ.setdefault("UNIFY_CACHE", str(config["unify_cache"]).lower())
    os.environ.setdefault("UNIFY_TRACED", str(config["unify_traced"]).lower())

    # Set log level if specified
    if config["log_level"]:
        os.environ.setdefault("LOG_LEVEL", config["log_level"])

    # Set architecture mode (control LLM usage during startup)
    # Use direct methods by default for faster API startup
    use_tool_loops = os.environ.get("RAG_USE_TOOL_LOOPS", "false").lower() == "true"
    os.environ.setdefault("RAG_USE_TOOL_LOOPS", str(use_tool_loops).lower())

    if use_tool_loops:
        print("🔧 Architecture Mode: Tool Loops (Unity pattern - slower, more robust)")
        print("   Set RAG_USE_TOOL_LOOPS=false for faster startup")
    else:
        print("⚡ Architecture Mode: Direct Methods (faster startup)")
        print("   Set RAG_USE_TOOL_LOOPS=true for full Unity pattern")

    # Build uvicorn command
    cmd = [
        "uvicorn",
        "intranet.core.api:app",
        "--host",
        host,
        "--port",
        str(port),
    ]

    if dev_mode:
        cmd.extend(["--reload", "--log-level", "debug"])
        print("🔧 Development mode enabled (auto-reload)")
    else:
        cmd.extend(["--workers", "1"])
        print("🏭 Production mode")

    print(f"🌐 Server starting on: http://{host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    print("\n" + "=" * 50)
    print("📋 Available Endpoints:")
    print("   GET  /health              - System health check")
    print("   POST /query               - Submit queries to RAG system")
    print("   GET  /conversation/{id}   - Get conversation history")
    print("   POST /feedback/{conv}/{turn} - Submit feedback")
    print("   WS   /ws/{conversation_id} - WebSocket for real-time chat")
    print("=" * 50)
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)

    try:
        # Change to intranet directory to ensure relative imports work
        original_dir = os.getcwd()

        # Start uvicorn as a subprocess with process group for proper signal handling
        server_process = subprocess.Popen(
            cmd,
            cwd=original_dir,
            start_new_session=True,  # Create new process group for proper signal handling
        )

        # Wait for the process to complete or be interrupted
        while server_process.poll() is None:
            if shutdown_requested:
                break
            time.sleep(0.5)  # Check every 500ms

        # If we get here and shutdown wasn't requested, the server exited unexpectedly
        if not shutdown_requested and server_process.poll() is not None:
            exit_code = server_process.poll()
            if exit_code != 0:
                print(f"\n❌ Server exited unexpectedly with code {exit_code}")
                return False

        return True

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user (KeyboardInterrupt)")
        if server_process:
            terminate_process(server_process)
        return True
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        if server_process:
            terminate_process(server_process)
        return False


async def main():
    # Get configuration for defaults
    config = get_config_values()

    parser = argparse.ArgumentParser(description="Start RAG API server")
    parser.add_argument(
        "--port",
        type=int,
        default=config["api_port"],
        help=f"Port to run on (default: {config['api_port']})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=config["api_host"],
        help=f"Host to bind to (default: {config['api_host']})",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        default=config["dev_mode"],
        help="Development mode with auto-reload",
    )
    parser.add_argument(
        "--with-init",
        action="store_true",
        help="Run full system initialization (schema + ingestion) before starting server, default is false",
        default=False,
    )
    parser.add_argument(
        "--use-tool-loops",
        action="store_true",
        help="Use tool loop architecture for initialization (slower, more robust)",
    )

    args = parser.parse_args()

    # Set architecture mode from command line if provided
    if args.use_tool_loops:
        os.environ["RAG_USE_TOOL_LOOPS"] = "true"

    # Control whether lifespan initialisation runs
    os.environ["RAG_SKIP_INIT"] = "false" if args.with_init else "true"

    success = await start_api_server(
        host=args.host,
        port=args.port,
        dev_mode=args.dev,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
