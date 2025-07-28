"""
Entry point.  GUI in main thread, Controller in background thread.
No Playwright code touches the Tk thread.
"""

import queue
import logging
import sys, pathlib

# Ensure repository root is on PYTHONPATH so `import unity` works when this
# script is executed directly from inside the "sandboxes" folder.
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(threadName)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("unity")

load_dotenv()

from unity.controller.gui import ControlPanel
from unity.controller.controller import Controller
from unity.controller.playwright_utils.worker import BrowserWorker


def main(use_controller: bool = True, debug: bool = True, mode: str = "hybrid") -> None:

    # queue for user commands only (GUI → backend)
    gui_to_backend_queue: queue.Queue[str] = queue.Queue(maxsize=50)

    if use_controller:
        # Use full Controller with act/observe capabilities and proper context management
        log.debug(f"Starting with full Controller (mode={mode}, debug={debug})...")
        redis_db = 0
        controller = Controller(
            session_connect_url=None,
            headless=False,
            mode=mode,
            debug=debug,
            redis_db=redis_db,
        )
        controller.start()

        # Redis publisher thread for commands - Controller listens on browser_command channel
        import redis, threading

        r = redis.Redis(host="localhost", port=6379, db=0)

        def _cmd_forwarder():
            while True:
                cmd = gui_to_backend_queue.get()
                # Commands go through redis to maintain compatibility with Controller's redis listener
                r.publish(f"browser_command_{redis_db}", cmd)

        threading.Thread(target=_cmd_forwarder, daemon=True).start()

        # launch Tk GUI (pulls browser_state directly from redis, sends commands via queue)
        gui = ControlPanel(gui_to_backend_queue, redis_db=redis_db)
        gui.set_controller(
            controller,
        )  # Give GUI access to Controller for act/observe calls

        try:
            gui.mainloop()
        finally:
            controller.stop()
            controller.join(timeout=2)

    else:
        # Use bare BrowserWorker for basic testing (no act/observe, potential context issues)
        log.debug(f"Starting with basic BrowserWorker (debug={debug})...")
        # queue for worker updates (worker → GUI)
        worker_to_gui_queue: queue.Queue[dict] = queue.Queue(maxsize=50)

        worker = BrowserWorker(
            commands_queue=gui_to_backend_queue,
            updates_queue=worker_to_gui_queue,
            headless=False,
            debug=debug,
            redis_db=redis_db,
        )
        worker.start()

        # launch Tk GUI
        gui = ControlPanel(gui_to_backend_queue, worker_to_gui_queue, redis_db=redis_db)
        gui.set_worker(worker)

        try:
            gui.mainloop()
        finally:
            worker.stop()
            worker.join(timeout=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unity Browser Controller Sandbox - Test browser automation with full Controller or basic BrowserWorker",
    )
    parser.add_argument(
        "--use-basic-worker",
        dest="use_controller",
        action="store_false",
        default=True,
        help="Use basic BrowserWorker instead of full Controller (disables act/observe methods and may cause context issues)",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=True,
        help="Enable debug mode (default: True)",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "vision", "heuristic"],
        default="hybrid",
        help="Controller mode: hybrid (default), vision, or heuristic",
    )

    args = parser.parse_args()
    main(use_controller=args.use_controller, debug=args.debug, mode=args.mode)
