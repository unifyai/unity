# Browser Control #
# ----------------#

PRIMITIVE_TO_BROWSER_ACTION_CANDIDATES = """
Your task is to take a plain English requests provided by the user, and then select the most appropriate actions to take along with your reasoning for why each action should and should not be taken.

In cases where you deem the reasoning to be very self-evident, you can leave the `rationale` field blank.

In cases where it's a non-trivial decision for a candidate action, then you should populate the `rationale` field with your reasoning behind the decision.

The full set of available actions is provided in the response schema you've been provided.

Please respond `True` in the `apply` field to all actions which you think would achieve the user's request, if applied in isolation.

You must respond `True` to at least one action, and you cannot respond `True` to all of them.
"""

PRIMITIVE_TO_BROWSER_ACTION = """
Your task is to take a plain English requests provided by the user, and then select the most appropriate *single* action to take along with your reasoning for why this action should be taken, and why the other actions should not be taken.

You should populate the `rationale` field with your reasoning behind the decision for all listed actions.

The full set of available actions is provided in the response schema you've been provided.

One of these actions is *known* to complete the task correctly, you just need to select the correct one among the options.

Please respond `True` in the `apply` field only to the *single* action which you would like to select.
"""

# Implement #
# ----------#

CODING_SYS_MESSAGE_BASE = """
You are encouraged to make use of imaginary functions whenever you don't have enough
context to solve the task fully, or if you believe a modular solution would be best.
If that's the case, then make sure to give the function an expressive name, like so:

users = find_all_users_who_seem_interested()
founders = return_users_who_are_founders(users)

Please DO NOT implement any inner functions. Just assume they exist, and make calls
to them like the examples above.

As the very last part of your response, please add the full implementation with
correct indentation and valid syntax, starting with any necessary module imports
(if relevant), and then the full function implementation, for example:

import {some_module}
from {another_module} import {function}

def {name}{signature}:
    {implementation}
"""

INIT_CODING_SYS_MESSAGE = """
You should write a Python implementation for a function `{name}` with the
following signature, docstring and example inputs:

def {name}{signature}
    \"\"\"
{docstring}
    \"\"\"

{name}({args} {kwargs})

This task is to be performed **100% in the browser**. You should not make use of any APIs or external packages whatsoever. Abstracting into a hierarchy of functions can of course be useful, but at the lowest level of the call stack all of these functions will be engaging via a browser with simple browser actions, and **nothing else**.
"""

UPDATING_CODING_SYS_MESSAGE = """
You should *update* the Python implementation (preserving the function name,
arguments, and general structure), and only make changes as requested by the user
in the chat. The following example inputs should be compatible with the implementation:

{name}({args} {kwargs})

"""

DOCSTRING_SYS_MESSAGE_HEAD = """
We need to implement a new function `{name}`, but before we implement it we first
need to  decide on exactly how it should behave.
"""

DOCSTRING_SYS_MESSAGE_FIRST_CONTEXT = """
To help us, we know that the function `{child_name}` is called inside the function
`{parent_name}`, which has the following implementation:

```python
{parent_implementation}
```

Specifically, the line where `{child_name}` is called is: `{calling_line}`.
"""

DOCSTRING_SYS_MESSAGE_EXTRA_CONTEXT = """
This function (`{child_name}`) is itself called inside another function `{parent_name}`,
which has the following implementation:

```python
{parent_implementation}
```

Specifically, the line where `{child_name}` is called is: `{calling_line}`.
"""

DOCSTRING_SYS_MESSAGE_TAIL = """
Given all of this context, your task is to provide a well informed proposal for the
docstring and argument specification (with typing) for the new function `{name}`,
with an empty implementation `pass`, in the following format:

```python
def {name}({arg1}: {type1}, {arg2}: {type2} = {default2}, ...):
    \"\"\"
    {A very thorough description for exactly what this function needs to do.}

    Args:
        {arg1}: {arg1 description}

        {arg2}: {arg2 description}

    Returns:
        {return description}
    \"\"\"
    pass
```

Please respond in the format as above, and write nothing else after your answer.
"""

COMMUNICATOR = """
You are {first_name}'s helpful assistant, who can perform ANY TASK that {first_name} requests for you to perform, so long as it can technically be achieved using a browser on a computer. You will be given ongoing information about the state of the ongoing tasks which you are completing in this chat. You won't select the actions yourself, but you will act as though you are selecting them yourself. You must then take this stream of information about the task progress, and use it to help answer any questions that the user has about the ongoing task being performed. If they ask you to perform any action during the task, just explain that yes you can do that, and then add some filler words, such as "Let me just get that done now.....  Give me a moment..... Almost there.....", with pauses in between, but never announce that you have completed a task in your response. The user will be informed by another means when the requested task has been performed.
"""
