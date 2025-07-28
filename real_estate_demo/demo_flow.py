import os

from dotenv import load_dotenv

load_dotenv()


from wizard_2 import (
    Node,
    Flow,
    InputField,
    RadioField,
    CheckBoxField,
    GoBack,
    GoNext,
    BaseGoToNode,
    EndCall,
    PromptUser,
    BaseDataFieldAction,
    # UpdateUser
)

from pydantic import BaseModel, Field
import openai

from datetime import datetime


def create_human_readable_delta(t):
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    delta = datetime.now() - t
    seconds = delta.seconds
    minutes = delta.seconds // 60
    if minutes:
        return f'{minutes} minute{"s" if minutes > 1 else ""} ago'
    else:
        return f'{"just now" if seconds <= 1 else str(seconds) + " seconds ago"}'


async def call_llm_3(
    sys: str,
    flow: Flow,
    conversation_history: list[str],
    action_log: list[str],
    model="gpt-4.1",
):
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    class AgentOutput(BaseModel):
        thoughts: str = Field(
            ...,
            description="Your inner thoughts before taking actions. Also determine if you need to give a small update to the user based on the conversation history",
        )
        # phone_utterance: Optional[str] = Field(...,
        # description="Your response to the user over the phone, shown as [Assistant] ... in the conversation history.")
        next_action: flow.current_action_model() | PromptUser | EndCall = Field(
            ...,
            description="next action to take given the current state.",
        )

    # print(flow.current_action_model().model_json_schema())
    # event_stream_str = "\n".join(event_stream)
    conversation_history_str = "\n".join(
        [
            f'[{m["role"].title()}, {create_human_readable_delta(m["timestamp"])}]: {m["message"]}'
            for m in conversation_history
        ],
    )
    conversation_history_prompt = (
        f"<conversation_history>\n{conversation_history_str}\n</conversation_history>"
    )

    action_log_str = "\n".join(
        [
            f'[{m["action"]}, {create_human_readable_delta(m["timestamp"])}]: {m["message"]}'
            for m in action_log
        ],
    )
    agent_script_prompt = f"""
<agent_script>
<action_log>
{action_log_str if action_log_str else 'No Actions Taken Yet'}
</action_log>

<current_node>
{flow.render()}
</current_node>
</agent_script>""".strip()
    user_msg = f"{conversation_history_prompt}\n\n{agent_script_prompt}"
    print("\033[32m" + user_msg + "\033[0m", flush=True)
    res = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": sys,
            },
            {
                "role": "user",
                "content": user_msg,
            },
        ],
        response_format=AgentOutput,
    )
    message = res.choices[0].message
    print(message)
    agent_output = message.parsed
    print(agent_output, flush=True)
    next_action = agent_output.next_action

    if isinstance(next_action, PromptUser):
        conversation_history.append(
            {
                "message": next_action.prompt,
                "role": "assistant",
                "timestamp": datetime.now(),
            },
        )

    else:

        if isinstance(next_action, BaseDataFieldAction):
            if next_action.update:
                conversation_history.append(
                    {
                        "message": next_action.update,
                        "role": "assistant",
                        "timestamp": datetime.now(),
                    },
                )
            flow.play_actions(next_action.fields_actions)
            next_action = next_action.fields_actions
        else:
            next_action = [next_action]
            flow.play_actions(next_action)
        # print(flow.current_node.title)
        action_events = []
        for action in next_action:
            if isinstance(action, EndCall):
                return
            elif isinstance(action, BaseGoToNode):
                action_event = f"went to node `{action.node_id}`"
                action_events.append((action, action_event))
            elif not isinstance(action, (GoNext, GoBack, PromptUser)):
                action_event = get_action_event(flow, action)
                action_events.append((action, action_event))
            else:
                if isinstance(action, GoNext):
                    action_event = (
                        f"advanced to the next node: '{flow.current_node.title}'"
                    )
                    action_events.append((action, action_event))
                elif isinstance(action, GoBack):
                    action_event = (
                        f"`went back to the previous node: '{flow.current_node.title}'"
                    )
                    action_events.append((action, action_event))
        for a, ae in action_events:
            action_log.append(
                {
                    "action": a.__class__.__name__,
                    "message": ae,
                    "timestamp": datetime.now(),
                },
            )
    return agent_output


def get_action_event(flow, action):
    print(flow.current_node.action_model, action.__class__)
    print(flow.current_node.action_model is action.__class__)
    field_id = flow.current_node.action_to_field[action.__class__]
    field = list(filter(lambda f: f.id == field_id, flow.current_node.fields))[0]
    if isinstance(field, InputField):
        return f"Input field '{field.label}' has been successfully filled with value: '{action.value}'"
    elif isinstance(field, RadioField):
        return f"Option '{action.value}' has been successfully selected for radio field '{field.label}'"
    elif isinstance(field, CheckBoxField):
        return f"Option {action.value}"


# flow = Flow([
#       start_call_screen,
#             profile_screen,
#              location_screen,
#              inside_home_area_screen,
#              floors_walls_stairs_screen,
#              ceiling_issues_screen,
#              cracks_in_the_ceiling_screen,
#              exact_location_screen,
#              confirmation_screen,
#              appointment_screen,
#              repair_ticket_raised_screen])
