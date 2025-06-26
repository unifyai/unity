import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


from wizard import Node, Flow, InputField, RadioField, CheckBoxField, GoBack, GoNext, EndSession

from pydantic import BaseModel, Field
import openai

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
    """Confirm with the tenant the repair ticket details by reading it out to them, and whether they would like to leave any additional notes.
Details to confirm with the user, in case they would like to change anything:
Location: {exact_location}
Area: {location}
Type: {area}
Issue: {area_tier_2} > {area_tier_2_issue}""".strip(),
fields=[
    RadioField(
        "confirm_repair_details",
        "Confirmeded Ticket Details?",
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

async def call_llm(sys: str, flow: Flow, event_stream: list, model="gpt-4.1"):
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    class AgentOutput(BaseModel):
        thoughts: str = Field(..., description="Your inner thoughts before responding or taking actions, your actions and response should be based on your thoughts")
        response: Optional[str] = Field(..., 
                                        description="Your response to the user, shown as [Assistant: ...] in the event stream, you can remain silent if simply navigating or taking mundane actions.")
        action: Optional[flow.current_action_model()] = Field(..., 
                                        description="action to take given the current state (state = events stream and agent script), shown as [Assistant took action `action_name`: ...] in the event_stream")
    
    # print(flow.current_action_model().model_json_schema())
    event_stream_str = "\n".join(event_stream)
    user_msg = f"<event_stream>\n{event_stream_str}\n</event_stream>\n\n<agent_script>\n{flow.render()}\n</agent_script>"
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
    if agent_output.response:
        event_stream.append(f"[Assistant: {agent_output.response}]")
    if agent_output.action:
        flow.play_actions(agent_output.action)
        # print(flow.current_node.title)
        for label, action in agent_output.action:
            if action is not None:
                if isinstance(EndSession):
                    return
                elif not isinstance(action, GoNext) and not isinstance(action, GoBack):
                    action_event = get_action_event(flow, action)
                else:
                    if isinstance(action, GoNext):
                        action_event = f"go_next and has advanced to the next node: '{flow.current_node.title}'"
                    else:
                        action_event = f"go_back and went back to the previous node: '{flow.current_node.title}'"
                event_stream.append(f"[Assistant took action `{label}`: {action_event}]")
                break
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