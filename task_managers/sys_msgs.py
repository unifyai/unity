_BASE = """
Your task is to first determine whether or not the **latest message** from the user contains an **explicit request** asking the assistant to either update or perform a task. The user-requested task might not be fully encapsulated in this latest message alone. For example, the latest user message in a live call transcript might just be: 'great, can you get started now then?', which is still a request despite not having all of the context. You should therefore consider the context of the **entire conversation history**, before deciding whether or not the latest message is requesting a task be started and/or updated. You should state the reasoning for your decision clearly in the reasoning field."
"""

FIRST_TASK = (
    _BASE
    + """
Following this, you must then create a new task based on this prior reasoning.

If no time information is provided, then you should assume the task is to be completed *now*, and that it is *not* a recurring task.

Please respond in the output format specified, stating your reasoning very clearly for each decision made.
"""
)

REORGANISE_TASKS = (
    _BASE
    + """
Following this, your must update the list of tasks based on this prior reasoning.

The current full set of tasks is as follows:

{current_tasks}

Please respond in the output format specified, stating your reasoning very clearly for each decision made.
"""
)
