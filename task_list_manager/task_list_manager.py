import threading


class TaskListManager(threading.Thread):

    def __init__(self, *, daemon: bool = True) -> None:
        """
        Responsible managing the list of tasks, updating the names, descriptions, schedles, repeating pattern and status of all tasks.
        """
        super().__init__(daemon=daemon)
        # ToDo: implement the tools
        self._tools = {}
