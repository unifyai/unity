"""Build service prompts for communication flows.
Replaces legacy .md files with programmatic builders."""

import inspect
import json
from typing import Dict, Callable


# Helpers for tool introspection
def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return a mapping from tool name to its signature."""
    return {name: str(inspect.signature(fn)) for name, fn in tools.items()}


def _tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Find the first tool whose name contains `needle` (case-insensitive)."""
    needle = needle.lower()
    return next((name for name in tools if needle in name.lower()), None)


# Shared prompt sections
def _build_event_stream_section(with_conductor: bool = True) -> str:
    """Build the Event Stream section with a heading and underline."""
    title = "Event Stream:"
    underline = "-" * len(title)
    items = [
        "You will be provided with a chronological event stream (may be truncated or partially omitted) containing the following types of events:",
        "1. User Message: Messages input by the user through the different communication channels like whatsapp, sms and email",
        "2. Assistant Message: Messages sent by you to the user through the different communication channels",
        "3. User and Assistant Phone Utterance: these are events emitted during phone calls, which are transcribed speech, can come from either party",
    ]
    if with_conductor:
        items.append(
            "4. Tasks: Tasks created through the Conductor and updates based on the handle actions.",
        )
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
        "- They also get logged into the event stream",
        # '- In case the user wants some information, use the "ask" action type otherwise if the user wants some action taken (such as scheduling tasks, sending mails, sms, etc.) use the "request" action type',
        "- You will be provided with a list of handles for all ongoing conductor tasks along with the query made to the conductor for each of them.",
        "- You should first check if there's an ongoing conductor task that the user is asking about or wants action taken on, before creating new conductor tasks",
        "- Never start a new task with the conductor if the user is asking you about an existing task!",
        "- In case the user wants action on an existing handle, use the conductor handle action with the appropriate handle action type and the handle id for the handle to be manipulated, along with the corresponding query",
        "- When a task is launched successfully, you should inform the user that you have started the task",
        "- Never, ever, make up names or numbers!",
    ]
    return "\n".join([title, underline] + rules)


def _build_communication_rules_section(with_conductor: bool = True) -> str:
    """Build the Communication Rules section with a heading and underline."""
    title = "Communication Rules:"
    underline = "-" * len(title)
    lines = [
        "- You are on a call with the user and should be mainly replying through the phone, unless sending messages via other channels makes sense.",
        "- Provide natural-sounding responses; the user expects human-like interactions.",
        "- Avoid verbose or unnecessary messages; only communicate when needed.",
        "- Maintain human-like language, avoid robotic and verbose responses",
        "- Do not overwhelm the user with useless messages or phone utterances, only send messages to the user when needed",
    ]
    if with_conductor:
        lines += [
            "- Any communication action (other than interactions on the current call) will happen through the conductor, so you'd need to create conductor tasks or act on existing tasks for any communication through whatsapp, sms, email, or sending a call.",
            "- Break large WhatsApp messages into multiple chunks when appropriate.",
            "- Send the full SMS message in one go when possible.",
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
            "- You are on an call with the user and can respond to the user through the phone alongside one of the communication channels (whatsapp, sms) through the Conductor",
            "- If you don't have the answer to the user's prompt, you should initiate a task using ConductorAction",
            "- If the user wants information or act on an existing task (you'd be provided with the currently ongoing tasks), you should use the ConductorHandleAction",
            "- You report back to the user the results of the Conductor task once it is done",
        ]
    else:
        lines = [
            "- You respond to the user through one of the communication channels (whatsapp, sms, phone) through the provided actions.",
            "- You can initiate communication tasks on the user's behalf by launching a communication task.",
            "- You report back to the user the results of communication task once they are done.",
        ]
    return "\n".join([title, underline] + lines)


# Refactored builders
def build_call_sys_prompt(name: str, with_conductor: bool = True) -> str:
    """Build the **system** prompt for phone-call LLM runs."""
    # assemble all sections
    sections = [
        "You are a general purpose AI assistant for your user.",
        _build_user_details_section(name),
        # _build_your_capabilities_section(is_call=True),
        _build_event_stream_section(with_conductor=with_conductor),
        _build_agent_loop_section() if with_conductor else "",
        _build_conductor_tasks_rules_section() if with_conductor else "",
        _build_communication_rules_section(with_conductor=with_conductor),
    ]
    # filter out None
    sections = [s for s in sections if s]
    return "\n\n".join(sections)


def build_non_call_sys_prompt(name: str, with_conductor: bool = True) -> str:
    """Build the **system** prompt for non-call LLM runs."""
    # assemble all sections
    sections = [
        "You are a general purpose AI assistant for your user.",
        _build_user_details_section(name),
        # _build_your_capabilities_section(is_call=False),
        _build_event_stream_section(with_conductor=with_conductor),
        _build_agent_loop_section() if with_conductor else "",
        _build_conductor_tasks_rules_section() if with_conductor else "",
        _build_communication_rules_section(with_conductor=with_conductor),
    ]
    sections = [s for s in sections if s]
    return "\n\n".join(sections)


def build_user_agent_prompt(
    call_purpose: str,
    past_events: list[dict],
    inflight_events: list[dict],
    conductor_handles: dict[int, dict] | None = None,
    with_conductor: bool = True,
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
        if conductor_handles and with_conductor
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
        (
            "** CONDUCTOR HANDLES (USE THESE FOR THE CONDUCTOR HANDLE ACTION) **"
            if with_conductor
            else ""
        ),
        conductor_handles_str.strip(),
    ]
    return "\n".join(lines)


def build_call_ask_prompt(tools: Dict[str, Callable], question: str) -> str:
    """Build the system prompt to await the user's reply and choose a tool."""
    # Dump tool signatures
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    local_chat_tool = _tool_name(tools, "local")
    ask_search_tool = _tool_name(tools, "user")

    # Assemble the ask prompt
    lines = [
        "Tools (name → argspec):",
        sig_json,
        "",
        "Search loop steps:",
        f"The question asked is '{question}'",
        f"First, find answer using `{local_chat_tool}`.",
        f"If an answer is not found, ask the user then search with `{ask_search_tool}`",
        f"If answer is not found, try `{local_chat_tool}` again.",
        f"If answer is still not found, then only select appropriate tools from all tools given.",
        "",
        "User's reply to the question asked in user message will be logged into the relevant managers. Run the appropriate tool to understand and return the user's answer.",
    ]
    return "\n".join(lines)


def build_action_prompt(tools: Dict[str, Callable], query: str) -> str:
    """Build the system prompt to await the user's reply and choose a tool."""
    # Dump tool signatures
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Assemble the ask prompt
    lines = [
        "Tools (name → argspec):",
        sig_json,
        "",
        f"Perform the query: {query} with the available tools above.",
    ]
    return "\n".join(lines)


def build_local_chat_search_prompt(local_chat_history: str) -> str:
    """Build the system prompt for searching the local chat history for an answer."""
    lines = [
        "The user is answering a question (given in user message).",
        "Local Chat History",
        "------------------",
        local_chat_history,
        "",
        "Search the chat history and summarise the answer if a response relevant to the question is found.",
        "Otherwise, return answer is not found.",
    ]
    return "\n".join(lines)
