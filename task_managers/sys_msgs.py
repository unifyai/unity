DETECT_TASK_REQUEST = """
Your task is to determine whether or not the provided messages contain a request from the user, asking the assistant to perform a task. You should state the reasoning for your decision clearly in the reasoning field, before responding with your final answer, `True` if a task was requested, and `False` if not."
"""

FIRST_TASK = """
Your task is to create a new task based on a conversation which *potentially* (not necessarily) contains instructions for a new task which needs to be completed.

If no time information is provided, then you should assume the task is to be completed *now*, and that it is *not* a recurring task.

Please respond in the output format specified, stating your reasoning very clearly for each decision made.

The conversation which may or may not contain information about a new task will be provided by the user.
"""

REORGANISE_TASKS = """
Your task is to update a list of tasks based on a conversation which *potentially* (not necessarily) contains instructions on which tasks to update, and how to update them.

The current full set of tasks is as follows:

{current_tasks}

The conversation which may or may not contain guidance for updating the tasks:

{conversation}

Please respond in the output format specified, stating your reasoning very clearly for each decision made.
"""
