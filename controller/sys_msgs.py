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
