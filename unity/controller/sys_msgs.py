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

PRIMITIVE_TO_BROWSER_MULTI_STEP = """
Your task is to take a plain English requests provided by the user, and then select the most appropriate *one* or *more* of action(s) to take along with your reasoning for why the actions should be taken, and why the other actions should not be taken.

You should populate the `rationale` field with your reasoning behind the decision for all listed actions.

One or several actions are *known* to complete the task correctly, you need to select the correct action(s) among the options.

List them in the exact order they should be executed.

Respond ONLY with valid JSON matching: {"rationale": "...", "action": "<prototype>", "value": <value|null>}

Available prototypes:
"""

PRIMITIVE_TO_BROWSER_ACTION_SIMPLE = """
You control the browser with ONE low‑level action.

Choose the best action‑prototype.

Respond ONLY with valid JSON matching: {"rationale": "...", "actions": ["<prototype>", ...]

Available prototypes:
"""