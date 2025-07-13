import sys

sys.path.append("..")
sys.path.append(".")


from wizard_2 import *

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
    next="area"
)

area_node = Node(
    "area",
    "Area",
    "Learn whether the issue is inside the home or outside",
    fields=[
        RadioField(
            "area",
            "Area",
            [
                "Inside Home",
                "Outside Home"
            ]
        )
    ],
    next="issue_section"
)

def get_type_node_fields(ctx):
    if ctx["area"] == "Inside Home":
        options = [
            "Floors, Walls, Ceilings and Stairs",
            "Plumbing",
            "Doors, Locks and Windows",
            "Electrics",
            "Alarms & Door Entry",
            "Heating & Hot Water",
            "Empty Repair"
        ]
    else:
        options = [
            "Gardens and Fences",
            "Plumbing",
            "Roofing",
            "Electrics"
        ]
    return [
        RadioField("section", "Section", options)
    ]
type_node = Node(
    "issue_section",
    "Issue Section",
    "Learn the Section that the issue is part of",
    fields = get_type_node_fields,
    next="issue_element"
)

    
from .issue_1 import get_issue_1_fields
issue_one_node = Node(
    "issue_element",
    "Issue Location",
    "Learn which exact element is affected",
    fields = get_issue_1_fields,
    next="issue_type"
)

from .dq import dq_options, get_dq_fields, handle_dq_next
from .issue_2 import get_issue_2_fields
issue_two_node = Node(
    "issue_type",
    "Issue type",
    "Learn which exact element is affected",
    fields=get_issue_2_fields,
    next=lambda ctx: "diagnostic_question" if ctx["issue_type"] in dq_options else "location" # either a diagnostic question then exact location or exact location directly
)


diagnostic_question_node = Node(
    "diagnostic_question",
    "Diagnostic Question",
    "Ask the user the following diagnostic question",
    fields=get_dq_fields,
    next=handle_dq_next
)

# diagnostic_question_2_node = Node(
#     "diagnostic_question_2",
#     "Diagnostic Question",
#     "Ask the user the following diagnostic question",
#     fields=...,
#     next="location"
# )


inside_options = ["Attic / Loft", "Bathroom", "Bedroom", "Cellar", "Dining room", "Hall", "Kitchen", "Landing", "Laundry room", "Living room", "Stairs"]
outside_options = ["External", "Garden", "Roof"]
location = Node(
    "location",
    "Issue Location",
    "Learn from the user the exact location of the issue",
    fields=lambda ctx: [RadioField("location", "Location", inside_options)] if ctx["area"] == "Inside Home" else [RadioField("location", "Location", outside_options)],
    next="confirmation"
)

confirmation_screen = Node(
    "confirmation",
    "Confirm Information",
    """Confirm with the tenant the repair ticket details before moving on to appointment reservation node, by reading it out to them, and whether they would like to leave any additional notes.
Details to confirm with the user, in case they would like to change anything:
Location: {location}
Area: {area}
Issue: {issue_element} > {issue_type}""".strip(),
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
next="appointment"
)

appointment_screen = Node(
    "appointment",
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
    """Steps: 
1. Inform the user one last time that a ticket with the following details has been raised.
Ticket Details:
Location: {location}
Area: {area}
Issue: {issue_element} > {issue_type}
Appointment Date: {chosen_slot}

2. Prompt the user if they need anything else
If they do not need anything:
    2.1 Thank the user and end the session
Else:
    2.2 Listen to their request and fulfill it if possible""".strip(),
fields=[RadioField("user_informed", "User has been informed one final time?", options=["yes"])],
next=None
)

diy_node = Node(
    "diy",
    "Do it yourself",
    """Steps:
1- Inform the tenant that this kind of repair request they'll need
to do themselves. they may want to contact
a local tradesperson or company if
they're unable to do this.

There is more information about who
carries out what repairs on our website
and in your tenancy agreement.
www.midlandheart.org.uk/repairtool

2. Prompt the user if they need anything else
If they do not need anything:
    2.1 Thank the user and end the session
Else:
    2.2 Listen to their request and fulfill it if possible""".strip(),
    fields=[RadioField("informed_diy", "Informed Tenant and tenant understood and accepted", options=["yes"])],
    next=None
)

def create_flow():
    return Flow([
        start_call_screen, profile_screen,
        area_node, type_node, issue_one_node, issue_two_node, diagnostic_question_node,
        location,
        confirmation_screen, 
        appointment_screen, 
        repair_ticket_raised_screen,
        diy_node
    ])