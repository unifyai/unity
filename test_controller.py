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

from controller.gui import ControlPanel
from controller.playwright.worker import BrowserWorker


def main() -> None:

    # queues
    gui_to_browser_queue: queue.Queue[str] = queue.Queue(maxsize=20)
    browser_to_gui_queue: queue.Queue[list] = queue.Queue(maxsize=20)
    llm_command_queue: queue.Queue[str] = queue.Queue(maxsize=100)

    # start worker thread
    worker = BrowserWorker(
        command_q=gui_to_browser_queue,
        update_q=browser_to_gui_queue,
        start_url="https://www.google.com/",
        refresh_interval=0.4,
        log=log.debug,
    )
    worker.start()

    # launch Tk GUI
    gui = ControlPanel(gui_to_browser_queue, browser_to_gui_queue, llm_command_queue)
    gui.set_worker(worker)

    try:
        gui.mainloop()
    finally:
        worker.stop()
        worker.join(timeout=2)


if __name__ == "__main__":
    main()
