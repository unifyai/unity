"""Build service prompts for communication flows.
Replaces legacy .md files with programmatic builders."""


# Shared prompt sections
def _build_event_stream_section() -> str:
    """Build the Event Stream section with a heading and underline."""
    title = "Event Stream:"
    underline = "-" * len(title)
    items = [
        "You will be provided with a chronological event stream (may be truncated or partially omitted) containing the following types of events:",
        "1. User Message: Messages input by the user through the different communication channels like whatsapp, sms and email",
        "2. Assistant Message: Messages sent by you to the user through the different communication channels",
        "3. User and Assistant Phone Utterance: these are events emitted during phone calls, which are transcribed speech, can come from either party",
        "4. Tasks: Tasks created and status updates",
    ]
    return "\n".join([title, underline] + items)


def _build_agent_loop_section() -> str:
    title = "Agent Loop:"
    underline = "-" * len(title)
    steps = [
        "You are operating in an agent loop, iteratively completing tasks through these steps:",
        "1. Analyze Events: Understand user needs and current state through event stream, focusing on latest user messages and tasks updates/statuses",
        "2. Select Action: Choose next action based on current state",
        "3. Async Actions: Actions are async by nature and results will not be immediately available, you will receive an event if an action was completed",
        "4. Iterate & Respond: You should repeat this loop (while responding to the user if deemed necessary)",
    ]
    return "\n".join([title, underline] + steps)


def _build_conductor_tasks_rules_section() -> str:
    title = "Conductor Tasks Rules:"
    underline = "-" * len(title)
    rules = [
        "- If the user asks about something that you can't answer based on the event history so far, you should use the conductor for performing it",
        "- Conductor actions launch a separate task in the background that you can keep track of in further steps",
        "- Any communication action (other than interactions on the current call) will happen through the conductor, so you'd need to create conductor tasks or act on existing tasks for any communication through whatsapp, sms, email, or sending a call.",
        "- They also get logged into the event stream",
        '- In case the user wants some information, use the "ask" action type otherwise if the user wants some action taken (such as scheduling tasks, sending mails, sms, etc.) use the "request" action type',
        "- You will be provided with a list of handles for all ongoing conductor tasks along with the query made to the conductor for each of them.",
        "- You should first check if there's an ongoing conductor task that the user is asking about or wants action taken on, before creating new conductor tasks",
        "- Never start a new task with the conductor if the user is asking you about an existing task!",
        "- In case the user wants action on an existing handle, use the conductor handle action with the appropriate handle action type and the handle id for the handle to be manipulated, along with the corresponding query",
        "- When a task is launched successfully, you should inform the user that you have started the task",
        "- Never, ever, make up names or numbers!",
    ]
    return "\n".join([title, underline] + rules)


def _build_communication_rules_section() -> str:
    """Build the Communication Rules section with a heading and underline."""
    title = "Communication Rules:"
    underline = "-" * len(title)
    lines = [
        "- You are on a call with the user and should be mainly replying through the phone, unless sending messages via other channels makes sense.",
        "- Provide natural-sounding responses; the user expects human-like interactions.",
        "- Break large WhatsApp messages into multiple chunks when appropriate.",
        "- Send the full SMS message in one go when possible.",
        "- Avoid verbose or unnecessary messages; only communicate when needed.",
        "- Maintain human-like language, avoid robotic and verbose responses",
        "- Do not overwhelm the user with useless messages or phone utterances, only send messages to the user when needed",
    ]
    return "\n".join([title, underline] + lines)


# Helper to build the user details section
def _build_user_details_section(name: str) -> str:
    title = "User Details:"
    underline = "-" * len(title)
    return "\n".join(
        [
            title,
            underline,
            f"User Name: {name}",
        ],
    )


# Helper to build the Your Capabilities section
def _build_your_capabilities_section(is_call: bool) -> str:
    """Build the Your Capabilities section with a heading and underline."""
    title = "Your Capabilities:"
    underline = "-" * len(title)
    if is_call:
        lines = [
            "- You are on a call with the user and can respond through the phone and any enabled channels.",
            "- Report back once your communication actions complete.",
        ]
    else:
        lines = [
            "- Respond to the user through one of the communication channels (whatsapp, sms, phone).",
            "- Report back once your communication actions complete.",
        ]
    return "\n".join([title, underline] + lines)


# Refactored builders
def build_call_sys_prompt(name: str, with_conductor: bool = True) -> str:
    """Build the **system** prompt for phone-call LLM runs.

    If `with_conductor` is True, include conductor task instructions."""
    # assemble all sections
    sections = [
        "You are a general purpose AI assistant for your user.",
        _build_user_details_section(name),
        _build_your_capabilities_section(is_call=True),
        _build_event_stream_section(),
        _build_agent_loop_section(),
        _build_conductor_tasks_rules_section() if with_conductor else None,
        _build_communication_rules_section(),
    ]
    # filter out None
    sections = [s for s in sections if s]
    return "\n\n".join(sections)


def build_non_call_sys_prompt(name: str, with_conductor: bool = True) -> str:
    """Build the **system** prompt for non-call LLM runs.

    If `with_conductor` is True, include conductor task instructions."""
    # assemble all sections
    sections = [
        "You are a general purpose AI assistant for your user.",
        _build_user_details_section(name),
        _build_your_capabilities_section(is_call=False),
        _build_event_stream_section(),
        _build_agent_loop_section(),
        _build_conductor_tasks_rules_section() if with_conductor else None,
        _build_communication_rules_section(),
    ]
    sections = [s for s in sections if s]
    return "\n\n".join(sections)


def build_user_agent_prompt(
    call_purpose: str,
    past_events: list[dict],
    inflight_events: list[dict],
    conductor_handles: dict[int, dict] | None = None,
) -> str:
    """Build the user-agent prompt including call purpose, events stream, and conductor handles."""
    from unity.conversation_manager.events import Event

    # Format past events
    past_events_str = (
        "\n".join(str(Event.from_dict(e)) for e in past_events) if past_events else ""
    )
    # Format new/inflight events
    new_events_str = (
        "\n".join(str(Event.from_dict(e)) for e in inflight_events)
        if inflight_events
        else ""
    )
    # Format conductor handles
    conductor_handles_str = (
        "\n".join(
            f"Handle ID {hid}: {conductor_handles[hid]['query']}"
            for hid in conductor_handles
        )
        if conductor_handles
        else ""
    )

    # Assemble lines for the prompt
    lines = [
        f"CALL PURPOSE: {call_purpose}",
        "Events Stream:",
        "** PAST EVENTS **",
        past_events_str.strip(),
        "** NEW EVENTS **",
        new_events_str.strip(),
        "** CONDUCTOR HANDLES (USE THESE FOR THE CONDUCTOR HANDLE ACTION) **",
        conductor_handles_str.strip(),
    ]
    return "\n".join(lines)


def build_ask_prompt(tool_names: list[str]) -> str:
    """Build a prompt for awaiting the user’s reply, listing available tools to act on their input."""
    title = "Ask Prompt:"
    underline = "-" * len(title)
    lines = [
        title,
        underline,
        "You have asked the user a question and are now awaiting their reply.",
        "Available tools:",
    ]
    # List each tool
    lines += [f"- {name}" for name in tool_names]
    lines += [
        "Once the user responds, choose and invoke the appropriate tool to return a summary of their response relevant to the question.",
    ]
    return "\n".join(lines)
