INTERJECTION_TO_BROWSER_ACTION = """
Your task is to take a plain English requests provided by the user, and then select the most appropriate action to take along with your reasoning for why each action should and should not be taken.

In cases where you deem the reasoning to be very self-evident, you can leave the field blank.

In cases where it's a non-trivial decision for a candidate action, then you should state the reasoning behind your decision in detail.

The full set of available actions is provided in the response schema you've been provided. Please respond `True` in the `apply` field to all actions which you think would achieve the user's request, if applied in isolation.

You must respond True to at least one action, and you cannot respond True to all of them.
"""
