import json
from typing import Dict

import unify


def tool_use_loop(client: unify.Unify, user_meessage: str, tools: Dict[str, callable]):
    """
    Loops the agent until no more tools are called, and the agent is satisfied.
    """
    while True:
        response = client.generate(user_meessage, return_full_completion=True)

        msg = response.choices[0].message
        if msg.tool_calls:
            # iterate over *all* tool calls returned in this turn
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)

                result = tools[name](**args)  # run the real function

                # feed result back so the model can think again
                client.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": result,
                    },
                )

            # also add the assistant placeholder (required for context)
            client.messages.append(msg)
            continue  # loop: model may ask for more tools

        else:
            # no tool_calls → model is satisfied
            break
