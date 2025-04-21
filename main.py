"""
Entry point.  GUI in main thread, BrowserWorker in background thread.
No Playwright code touches the Tk thread.
"""

import queue
import logging

from dotenv import load_dotenv  # pip install python-dotenv

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(threadName)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("unity")

load_dotenv()

from gui import ControlPanel
from worker import BrowserWorker
from primitive import init as primitive_init


# unify.activate("Unity", overwrite=True)


def run_tasks():
    pass
    # from tasks.log_into_gmail import log_into_gmail
    # log_into_gmail()


# @unify.traced
def main() -> None:
    # queues
    cmd_q: "queue.Queue[str]" = queue.Queue(maxsize=20)
    up_q: "queue.Queue[list]" = queue.Queue(maxsize=20)
    text_q: queue.Queue[str] = queue.Queue(maxsize=100)

    # start worker thread
    worker = BrowserWorker(
        cmd_q,
        up_q,
        start_url="https://www.google.com/",
        refresh_interval=0.4,
        log=log.debug,
    )
    worker.start()

    # launch Tk GUI
    gui = ControlPanel(cmd_q, up_q, text_q)
    gui.set_worker(worker)
    primitive_init(text_q.put)

    gui.after(1000, run_tasks)

    try:
        gui.mainloop()
    finally:
        worker.stop()
        worker.join(timeout=2)


if __name__ == "__main__":
    main()
