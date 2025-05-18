"""
Entry point.  GUI in main thread, BrowserWorker in background thread.
No Playwright code touches the Tk thread.
"""

import queue
import logging

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(threadName)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("unity")

load_dotenv()

from unity.controller.gui import ControlPanel
from unity.controller.playwright.worker import BrowserWorker


def main() -> None:

    # queue for user commands only (GUI → redis)
    gui_to_browser_queue: queue.Queue[str] = queue.Queue(maxsize=50)

    # Start BrowserWorker (publishes browser_state on redis)
    worker = BrowserWorker(
        start_url="https://www.google.com/",
        refresh_interval=0.4,
        log=log.debug,
    )
    worker.start()

    # Redis publisher thread for commands
    import redis, threading

    r = redis.Redis(host="localhost", port=6379, db=0)

    def _cmd_forwarder():
        while True:
            cmd = gui_to_browser_queue.get()
            r.publish("browser_command", cmd)

    threading.Thread(target=_cmd_forwarder, daemon=True).start()

    # launch Tk GUI (pulls browser_state directly from redis)
    gui = ControlPanel(gui_to_browser_queue)

    try:
        gui.mainloop()
    finally:
        worker.stop()
        worker.join(timeout=2)


if __name__ == "__main__":
    main()
