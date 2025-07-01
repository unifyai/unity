import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


from wizard import (
    Node, Flow, InputField, RadioField, 
CheckBoxField, GoBack, GoNext, BaseGoToNode ,EndSession, PromptUser, BaseDataFieldAction
# UpdateUser
)

from pydantic import BaseModel, Field
import openai

from datetime import datetime, timedelta

def create_human_readable_delta(t):
    delta = datetime.now() - t
    seconds = delta.seconds
    minutes = delta.seconds // 60
    if minutes:
        return f'{minutes} minute{"s" if minutes > 1 else ""} ago'
    else:
        return f'{"just now" if seconds <= 1 else str(seconds) + " seconds ago"}'

start_call_screen = Node(
    "start_call_screen",
    "Call Start",
    instructions="""Greet the user, Inroduce yourself, then Learn what kind of service does the user need, raise a repair ticket, or update an existing one.""",
    fields=[
        RadioField("service_type", "Service Type", ["Raise repair ticket", "Update existing ticket"])
    ],
    next={
        "Raise repair ticket": "profile_screen",
        "Update existing ticket": ...
    }
)

profile_screen = Node(
    "profile_screen",
    "Profile",
    instructions="""Steps to perform:
1- Ask the user about their issue (let them describe their problem)
2- Then ask the user for their name and address to start the flow.""",
    fields=[
        InputField("tenant_name", "Tenant Name"),
        InputField("tenant_address", "Tenant Address")
    ],
    next="location_screen"
)

location_screen = Node(
    "location_screen",
    "Location",
    instructions="""Learn whether the issue is inside their home or outside""",
    fields=[
        RadioField("location", "Location", [
            "Inside home",
            "Outside home"
        ])
    ],
    next=lambda ctx: "inside_home_area_screen" if ctx["location"] == "Inside home" else "outside_home_area_screen"
)


inside_home_area_screen = Node(
    "inside_home_area_screen",
    "Inside Home Area",
    instructions="Ask the user about the area where the issue is",
    fields=[
        RadioField("area", "Area", [
            "Floors, Walls, Ceilings and Stairs",
            "Plumbing",
            "Doors, Locks and Windows",
            "Electrics",
            "Alarms & Door Entry",
            "Heating & Hot Water",
            "Empty Repair"
        ])
    ],
    next={
        "Floors, Walls, Ceilings and Stairs": "floors_walls_stairs_screen",
        "Plumbing": "plumbing_screen",
        # "Roof leaking": "roof_leaking_screen"
    }
)

floors_walls_stairs_screen = Node(
    "floors_walls_stairs_screen",
    "Area: Floors, Walls, Ceilings and Stairs",
    "Ask the user about the precise area where the issue is",
    fields=[
        RadioField(
            "area_tier_2",
            "Which area excatly has the problem?",
            [
                "Floors",
                "Ceilings",
                "Walls",
                "Stairs"
            ]
        )
    ],
    next={
        "Ceilings": "ceilings_issues_screen",
        "Floors": "floor_issues_screen"
    }
)

ceiling_issues_screen = Node(
    "ceilings_issues_screen",
    "Ceilings Issues",
    "Learn from the user which kind of ceiling issue they area dealing with",
    fields=[
        RadioField("area_tier_2_issue", "Ceiling Issue", [
            "Ceiling is falling down",
            "Cracks in the ceiling",
            "Roof leaking"
    ])
    ],
    next={
        "Ceiling is falling down": "ceiling_is_falling_down_screen",
        "Cracks in the ceiling": "cracks_in_the_ceiling_screen",
        "Roof leaking": "roof_leaking_screen"
    }
)

cracks_in_the_ceiling_screen = Node(
    "cracks_in_the_ceiling_screen",
    "Cracks in the ceiling",
    "Learn whether the crack can fit a 1 euro coin or not",
    fields=[
        RadioField("crack_can_fit_coin", "Could you fit a one euro coin in the gap?", 
                   [
                       "yes",
                       "no"
                   ])
    ],
    next="exact_location_screen"
)

exact_location_screen = Node(
    "exact_location_screen",
    "Exact Location",
    "Learn the exact location of the issue",
    fields=[
        RadioField("exact_location", "Exact Location/Room of Issue", options=[
            "Attic",
            "Kitchen",
            "Bathroom",
            "Hall",
            "Laundry Room",
            "Bedroom",
            "Dining Room"
        ])
    ],
    next="confirmation_screen"
)

confirmation_screen = Node(
    "confirmation_screen",
    "Confim Information screen",
    """Confirm with the tenant the repair ticket details before moving on to appointment reservation node, by reading it out to them, and whether they would like to leave any additional notes.
Details to confirm with the user, in case they would like to change anything:
Location: {exact_location}
Area: {location}
Type: {area}
Issue: {area_tier_2} > {area_tier_2_issue}""".strip(),
fields=[
    RadioField(
        "confirm_repair_details",
        "Confirmed Ticket Details?",
        options=["Yes"]
    ),
    InputField(
        "additional_notes",
        "Additional Notes by Tenant",
        required=False
    )
],
next="appointment_screen"
)

appointment_screen = Node(
    "appointment_screen",
    "Appointment Reservation",
    "Inform the user about the available time slots for a repair technician to visit",
    fields=[
        RadioField("chosen_slot", "Available slots", 
                   options=[          
                    "Mon 10 Feb 2025, 8:00 AM TO 1:00 PM",
                    "Mon 10 Feb 2025, 8:00 AM TO 5:00 PM",
                    "Mon 10 Feb 2025, 9:30 AM TO 1:30 PM",
                    "Mon 10 Feb 2025, 12:00 PM TO 5:00 PM",
                    "Tue 11 Feb 2025, 8:00 AM TO 1:00 PM",
                    "Tue 11 Feb 2025, 8:00 AM TO 5:00 PM",
                    "Tue 11 Feb 2025, 9:30 AM TO 1:30 PM",
                    "Tue 11 Feb 2025, 12:00 PM TO 5:00 PM"
                   ])
    ],
    next="repair_ticket_raised_screen"
)

repair_ticket_raised_screen = Node(
    "repair_ticket_raised_screen",
    "Repair Ticket Successfully Raised",
    """Inform the user that a ticket with the following details has been raised. After that ask them, if they need anything else, if not thank them and end the session.
Ticket Details:
Location: {exact_location}
Area: {location}
Type: {area}
Issue: {area_tier_2} > {area_tier_2_issue}
Appointment Date: {chosen_slot}""".strip(),
fields=[RadioField("user_informed", "User has been informed and has consented?", options=["yes"])],
next=None
)

async def call_llm(sys: str, flow: Flow, conversation_history: list[str], action_log: list[str], model="gpt-4.1"):
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    class AgentOutput(BaseModel):
        thoughts: str = Field(..., description="Your inner thoughts before taking actions.")
        # phone_utterance: Optional[str] = Field(..., 
                                        # description="Your response to the user over the phone, shown as [Assistant] ... in the conversation history.")
        action: Optional[flow.current_action_model()] = Field(..., 
                                        description="action to take given the current state.")
    
    # print(flow.current_action_model().model_json_schema())
    # event_stream_str = "\n".join(event_stream)
    conversation_history_str = "\n".join([f'[{m["role"].title()}, {create_human_readable_delta(m["timestamp"])}]: {m["message"]}' for m in conversation_history])
    conversation_history_prompt = f'<conversation_history>\n{conversation_history_str}\n</conversation_history>'

    action_log_str = "\n".join([f'[{m["action"]}, {create_human_readable_delta(m["timestamp"])}]: {m["message"]}' for m in action_log])
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
    if agent_output.phone_utterance:
        conversation_history.append({"message": agent_output.phone_utterance, "role": "assistant", "timestamp": datetime.now()})
    if agent_output.action:
        flow.play_actions(agent_output.action)
        action = agent_output.action
        # print(flow.current_node.title)
        
        if action is not None:
            if isinstance(action, EndSession):
                return
            elif isinstance(action, BaseGoToNode):
                action_event = f"went to node `{action.node_id}`"
            elif not isinstance(action, GoNext) and not isinstance(action, GoBack):
                action_event = get_action_event(flow, action)
            else:
                if isinstance(action, GoNext):
                    action_event = f"advanced to the next node: '{flow.current_node.title}'"
                else:
                    action_event = f"`went back to the previous node: '{flow.current_node.title}'"
            action_log.append({"action": action.__class__.__name__, "message": action_event, "timestamp": datetime.now()})
    return agent_output


async def call_llm_3(sys: str, flow: Flow, conversation_history: list[str], action_log: list[str], model="gpt-4.1"):
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    class AgentOutput(BaseModel):
        thoughts: str = Field(..., description="Your inner thoughts before taking actions. Also determine if you need to give a small update to the user based on the conversation history")
        # phone_utterance: Optional[str] = Field(..., 
                                        # description="Your response to the user over the phone, shown as [Assistant] ... in the conversation history.")
        next_action: flow.current_action_model() | PromptUser | EndSession = Field(..., 
                                        description="next action to take given the current state.")
    
    # print(flow.current_action_model().model_json_schema())
    # event_stream_str = "\n".join(event_stream)
    conversation_history_str = "\n".join([f'[{m["role"].title()}, {create_human_readable_delta(m["timestamp"])}]: {m["message"]}' for m in conversation_history])
    conversation_history_prompt = f'<conversation_history>\n{conversation_history_str}\n</conversation_history>'

    action_log_str = "\n".join([f'[{m["action"]}, {create_human_readable_delta(m["timestamp"])}]: {m["message"]}' for m in action_log])
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
        conversation_history.append({"message": next_action.prompt, "role": "assistant", "timestamp": datetime.now()})

    else:
        
        if isinstance(next_action, BaseDataFieldAction):
            if next_action.update:
                conversation_history.append({"message": next_action.update, "role": "assistant", "timestamp": datetime.now()})
            flow.play_actions(next_action.fields_actions)
            next_action = next_action.fields_actions
        else:
            next_action = [next_action]
            flow.play_actions(next_action)
        # print(flow.current_node.title)
        action_events = []
        for action in next_action:
            if isinstance(action, EndSession):
                return
            elif isinstance(action, BaseGoToNode):
                action_event = f"went to node `{action.node_id}`"
                action_events.append((action, action_event))
            elif not isinstance(action, (GoNext, GoBack, PromptUser)):
                action_event = get_action_event(flow, action)
                action_events.append((action, action_event))
            else:
                if isinstance(action, GoNext):
                    action_event = f"advanced to the next node: '{flow.current_node.title}'"
                    action_events.append((action, action_event))
                elif isinstance(action, GoBack):
                    action_event = f"`went back to the previous node: '{flow.current_node.title}'"
                    action_events.append((action, action_event))
        for a, ae in action_events:
            action_log.append({"action": a.__class__.__name__, "message": ae, "timestamp": datetime.now()})
    return agent_output


async def call_llm_2(sys: str, flow: Flow, conversation_history: list[str], action_log: list[str], model="gpt-4.1"):
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    class AgentOutput(BaseModel):
        thoughts: str = Field(..., description="Your inner thoughts before responding or taking actions, your actions and response should be based on your thoughts")
        response: Optional[str] = Field(..., 
                                        description="Your response to the user, shown as [Assistant] ... in the conversation history.")
        action: Optional[flow.current_action_model()] = Field(..., 
                                        description="action to take given the current state.")
    
    # print(flow.current_action_model().model_json_schema())
    # event_stream_str = "\n".join(event_stream)
    conversation_history_msgs = [{"role": m["role"], "content": f'[{m["role"].title()}, {create_human_readable_delta(m["timestamp"])}]: {m["message"]}'} for m in conversation_history]
    action_log_str = "\n".join([f'[{m["action"]}, {create_human_readable_delta(m["timestamp"])}]: {m["message"]}' for m in action_log])
    agent_script_prompt = f"""
<agent_script>
<action_log>
{action_log_str if action_log_str else 'No Actions Taken Yet'}
</action_log>

<current_node>
{flow.render()}
</current_node>
</agent_script>""".strip()
    user_msg = f"{agent_script_prompt}"
    print("\033[32m" + str(conversation_history_msgs) + "\033[0m", flush=True)
    print("\033[32m" + user_msg + "\033[0m", flush=True)
    res = await client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": sys,
                    },
                    *conversation_history_msgs,
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
    if agent_output.response:
        conversation_history.append({"message": agent_output.response, "role": "assistant", "timestamp": datetime.now()})
    if agent_output.action:
        flow.play_actions(agent_output.action)
        action = agent_output.action
        # print(flow.current_node.title)
        
        if action is not None:
            if isinstance(action, EndSession):
                return
            elif isinstance(action, BaseGoToNode):
                action_event = f"went to node `{action.node_id}`"
            elif not isinstance(action, GoNext) and not isinstance(action, GoBack):
                action_event = get_action_event(flow, action)
            else:
                if isinstance(action, GoNext):
                    action_event = f"advanced to the next node: '{flow.current_node.title}'"
                else:
                    action_event = f"`went back to the previous node: '{flow.current_node.title}'"
            action_log.append({"action": action.__class__.__name__, "message": action_event, "timestamp": datetime.now()})
    return agent_output


def get_action_event(flow, action):
    field_id = flow.current_node.action_to_field[action.__class__]
    field = list(filter(lambda f: f.id == field_id, flow.current_node.fields))[0]
    if isinstance(field, InputField):
        return f"Input field '{field.label}' has been successfully filled with value: '{action.value}'"
    elif isinstance(field, RadioField):
        return f"Option '{action.value}' has been successfully selected for radio field '{field.label}'"
    elif isinstance(field, CheckBoxField):
        return f"Option {action.value}"



flow = Flow([
      start_call_screen,
            profile_screen, 
             location_screen, 
             inside_home_area_screen, 
             floors_walls_stairs_screen, 
             ceiling_issues_screen, 
             cracks_in_the_ceiling_screen,
             exact_location_screen,
             confirmation_screen,
             appointment_screen,
             repair_ticket_raised_screen])