"""
Entry point.  GUI in main thread, BrowserWorker in background thread.
No Playwright code touches the Tk thread.
"""

import queue

from dotenv import load_dotenv  # pip install python-dotenv

load_dotenv()

from gui import ControlPanel
from worker import BrowserWorker

START_URL = "https://unify.ai"


def main() -> None:
    # queues
    cmd_q: "queue.Queue[str]" = queue.Queue(maxsize=20)
    up_q: "queue.Queue[list]" = queue.Queue(maxsize=20)

    # start worker thread
    worker = BrowserWorker(cmd_q, up_q, start_url=START_URL, refresh_interval=0.4)
    worker.start()

    # launch Tk GUI
    gui = ControlPanel(cmd_q, up_q)
    try:
        gui.mainloop()
    finally:
        worker.stop()
        worker.join(timeout=2)


if __name__ == "__main__":
    main()
