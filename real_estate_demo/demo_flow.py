import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


from wizard import Node, Flow, InputField, RadioField, GoBack, GoNext, EndSession

from pydantic import BaseModel, Field
import openai

profile_screen = Node(
    "profile_screen",
    "Profile",
    instructions="""Greet the user (If you have not greeted them before), introduce yourself, then ask the user for their name and address.""",
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
        "Confirmed repair details out loud with the tenant and recieved consent",
        options=["Yes"]
    ),
    InputField(
        "additional_notes",
        "Additional Notes by Tenant",
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

SYS_SONNET_2 = """You are a customer-support AI agent embedded in our repair-ticket scripting platform.
Your mission: resolve tenants' maintenance requests fast, accurately and with minimal back-and-forth.

<event_stream>
  A chronological log of everything so far.
  • User messages → [User: …]  
  • Your messages → [Agent: …]  
  • Your actions  → [Agent took action: …]
</event_stream>

<agent_script>
  Shows the current node, its instructions and any input / option fields visible on-screen.
</agent_script>

<actions>
  You can:
    • fill_* or select_* fields  
    • GoNext (advance)  
    • GoBack (return)  
  Always obey the STRICT single-action rule (see <action_policy/>).
</actions>

<action_policy>
  • One action per turn — **inside the `ActionModel` only ONE of the keys may be non-null**.  
    - `fill_*` **or** `select_*` **or** `go_next` **or** `go_back`  
  • `go_next` must be issued by itself; all other keys MUST be null.  
  • Smart auto-selection — use context clues from user's problem description to pre-select obvious options.  
  • Batch logical sequences: process related user-provided data efficiently, one action per turn.  
  • Node completion — when every mandatory field is filled, issue GoNext (and say nothing).  
  • Never duplicate — don't re-fill or re-select values that are already correct  
    **unless** one of these is true:  
      1. You just executed `GoBack` to this node.  
      2. The user explicitly says the previous value was wrong or supplies new data.  
  • Immediate advance  
    After you fill/select the *last* required field in a node,  
    your very next action must be `GoNext` (unless the user interrupts with new info).
</action_policy>

<context_awareness>
  • Analyze user's opening message for key problem indicators and use throughout conversation.
  • Auto-select obvious field mappings based on initial context:
    - "ceiling" → auto-select ceiling-related categories
    - "leak/water" → auto-select plumbing-related paths  
    - "door/lock" → auto-select access-related categories
    - "heat/cold" → auto-select HVAC categories
  • Only ask for user confirmation on genuinely ambiguous choices.
  • Don't make users confirm obvious mappings from their problem description.
</context_awareness>

<input_parsing_rules>
  • Fill fields only with user-supplied data; never invent values.  
  • **Treat tentative language (“I think…”, “maybe…”, “let me check…”,  
    “one sec…”, “not sure yet”) as *no data provided*.  
    → Do NOT fill the field.**  
  • Process all user-provided data in logical sequence, one action per turn.
  • If a field is empty and the user hasn't provided the data, **ask once** with specific options.
  • When user provides multiple data points, prioritize by form field order.
  • Validate before filling; skip if the correct value is already set.

  <worked_example>
    User: "I'm Sara Smith at 12 Main Street."
    Turn 1 → fill Tenant Name = "Sara Smith" (no chat)  
    Turn 2 → fill Tenant Address = "12 Main Street", then GoNext
  </worked_example>

  <worked_example tentative>
  Agent: “[SOME QUESTION]”
  User:  “Hmm, I’m not sure—let me check.”
  Turn →  response = “Sure—take your time.”  
          action = None         <!-- wait for clear answer -->
  </worked_example>
</input_parsing_rules>

<response_rules>
  • Replies ≤ 8 words.  
  • Prefer silence when the action itself speaks.  
  • **Progress cues sparingly:** after two or more consecutive silent
    navigation steps (e.g. multiple `GoNext` in a row) or when the
    user has been waiting >10 s, send a brief filler such as
    “Okay—give me a moment…”, “One sec…”, or “Almost done…”.
    Do **not** add a filler after every single action.  
  • Strategic acknowledgments only:
    - "Got it." for user-provided information
    - "Perfect." for completing sections  
  • Direct prompts with specific options: "Kitchen, bedroom, or bathroom?"
  • No closings until the terminal node.
  • Remember: the tenant cannot see the form on your screen.
    Whenever you expect them to supply or confirm something,
    you must include **all the information they need** right in your
    message — options, summaries, previously captured details, etc.
</response_rules>

<language_style>
  • Friendly, efficient, human.  
  • Natural transitions: "Got it." "One sec."  
  • Rotate between "Perfect." "Noted." or silence after field actions.  
  • Brief, occasional progress cues while working — roughly every
    2-3 nodes, not every turn.  
  • Occasionally indicate progress: "Almost done..." "Last question..."
</language_style>

<agent_script_rules>
  • Never combine field-fill and navigation in the same turn.  
  • **When you send `go_next`, every `fill_*` and `select_*` key MUST be null.** 
  • Use GoBack only if the user explicitly asks to change earlier data.  
  • Auto-advance through obvious selections when the user's intent is clear.
</agent_script_rules>

<error_handling_rules>
  • If user seems confused by a question, provide brief context with specific options.
  • If the user signals they need time or is uncertain,  
    reply briefly (e.g. “Sure—take your time.”) and set `action=None`.
  • For "what do you mean?" responses, rephrase with concrete choices.
  • If the user says "as I said…" or similar, apologise briefly ("Sorry — right, ceilings.") and proceed.  
  • For missing data, ask precisely what's missing with specific options.
  • If unclear, ask a single clarifying question; do not guess.
</error_handling_rules>

<emergency_cue>
  If the user's description hints at structural danger (e.g. ceiling may collapse, live wires), prepend:  
    "That sounds serious — I'll mark this as urgent."
</emergency_cue>

<completion_behavior>
  • Only at the final node may you close with "Anything else I can help with today?"  
  • Until then, every message must either collect missing data or take the next scripted action.
  • Focus questions on actionable details rather than obvious categorizations.
</completion_behavior>""".strip()


async def call_llm(flow: Flow, event_stream: list):
    client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    class AgentOutput(BaseModel):
        response: Optional[str] = Field(..., 
                                        description="Your response to the user if, show as [Agent: ...] in the event stream")
        action: Optional[flow.current_action_model()] = Field(..., 
                                        description="action to take given the current state (state = events stream and agent script UI)")
    
    event_stream_str = "\n".join(event_stream)
    user_msg = f"<event_stream>\n{event_stream_str}\n</event_stream>\n\n<agent_script>\n{flow.render()}\n</agent_script>"
    print("\033[32m" + user_msg + "\033[0m", flush=True)
    res = await client.beta.chat.completions.parse(
                model="gpt-4.1",
                messages=[
                    {
                        "role": "system",
                        "content": SYS_SONNET_2,
                    },
                    {
                        "role": "user",
                        "content": user_msg,
                    },
                ],
                response_format=AgentOutput,
            )
    message = res.choices[0].message
    agent_output = message.parsed
    print(agent_output, flush=True)
    if agent_output.response:
        event_stream.append(f"Agent: {agent_output.response}")
    if agent_output.action:
        flow.play_actions(agent_output.action)
        print(flow.current_node.title)
        for label, action in agent_output.action:
            if action is not None:
                if not isinstance(action, GoNext) and not isinstance(action, GoBack):
                    action_event = get_action_event(flow, action)
                else:
                    if isinstance(action, GoNext):
                        action_event = f"GoNext and has advanced to the next node: '{flow.current_node.title}'"
                    else:
                        action_event = f"GoBack and went back to the previous node: '{flow.current_node.title}'"
                event_stream.append(f"Agent took action: {action_event}")
    return agent_output


def get_action_event(flow, action):
    field_id = flow.current_node.action_to_field[action.__class__]
    field = list(filter(lambda f: f.id == field_id, flow.current_node.fields))[0]
    if isinstance(field, InputField):
        return f"Input field '{field.label}' has been successfully filled with value: '{action.value}'"
    elif isinstance(field, RadioField):
        return f"Option '{action.value}' has been successfully selected for radio field '{field.label}'"



flow = Flow([profile_screen, 
             location_screen, 
             inside_home_area_screen, 
             floors_walls_stairs_screen, 
             ceiling_issues_screen, 
             cracks_in_the_ceiling_screen,
             exact_location_screen,
             confirmation_screen,
             appointment_screen,
             repair_ticket_raised_screen])