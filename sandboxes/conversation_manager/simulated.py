import asyncio
import unify
from sandboxes.utils import (
    build_cli_parser,
    activate_project,
    speak,
    record_until_enter,
    transcribe_deepgram,
)
from unity.conversation_manager.prompt_builders import (
    build_call_sys_prompt,
    build_user_agent_prompt,
)
from unity.conversation_manager.events import (
    PhoneUtteranceEvent,
    SMSMessageRecievedEvent,
    SMSMessageSentEvent,
    EmailRecievedEvent,
    EmailSentEvent,
)


MEDIUM = "phone"

RECEIVED_MAP = {
    "phone": PhoneUtteranceEvent,
    "sms": SMSMessageRecievedEvent,
    "email": EmailRecievedEvent,
}

SENT_MAP = {
    "phone": PhoneUtteranceEvent,
    "sms": SMSMessageSentEvent,
    "email": EmailSentEvent,
}

COMMANDS_HELP = (
    "\nAgent-User simulation – type commands below. 'exit' to end.\n\n"
    "┌───────────────────────────────── accepted commands ─────────────────────────────────┐\n"
    "│ start (s)      – begin conversation                                                 │\n"
    "│ continue (c)   – continue for next 3 exchanges                                      │\n"
    "│ medium (m)     – change communication medium (restarts conversation with history)   │\n"
    "│ help (h)       – show this help                                                     │\n"
    "│ exit | quit    – end simulation                                                     │\n"
    "└─────────────────────────────────────────────────────────────────────────────────────┘"
)


def make_agent(
    job,
    user_name,
    assistant_name,
    assistant_number,
    user_number,
    assistant_behaviour,
):
    client = unify.AsyncUnify(endpoint="gpt-4.1@openai")
    return {
        "job": job,
        "user_name": user_name,
        "assistant_name": assistant_name,
        "assistant_age": "30",
        "assistant_region": "US",
        "assistant_about": assistant_behaviour,
        "assistant_number": assistant_number,
        "user_number": user_number,
        "purpose": assistant_behaviour,
        "task_context": None,
        "history": [],
        "client": client,
    }


async def simulate_turn(agent, message, start=False):
    global MEDIUM
    if not start:
        # record user utterance via appropriate event class
        recv_cls = RECEIVED_MAP.get(MEDIUM, PhoneUtteranceEvent)
        ue = recv_cls(timestamp=None, content=message, role="User")
        agent["history"].append(ue.to_dict())
    # build system prompt
    sys_prompt = build_call_sys_prompt(
        agent["user_name"],
        agent["assistant_name"],
        agent["assistant_age"],
        agent["assistant_region"],
        agent["assistant_about"],
        agent["task_context"],
        broader_context="",
    )
    # build user-agent prompt
    ua_prompt = build_user_agent_prompt(
        call_purpose=agent["purpose"],
        past_events=agent["history"],
        inflight_events=agent["history"][-1:],
        tool_use_handles=None,
    )
    # run real AsyncUnify
    client = agent["client"]
    client.set_system_message(sys_prompt)
    # single-turn generate
    resp = await client.generate(user_message=ua_prompt)
    # record assistant response via appropriate event class
    reply = getattr(resp, "phone_utterance", None) or str(resp)
    send_cls = SENT_MAP.get(MEDIUM, PhoneUtteranceEvent)
    ae = send_cls(timestamp=None, content=reply, role="Assistant")
    agent["history"].append(ae.to_dict())
    return reply


async def simulate():
    global MEDIUM
    parser = build_cli_parser("Raw two-agent simulation")
    args = parser.parse_args()
    activate_project(args.project_name, args.overwrite)

    # Input for customised user profile
    print("Configure Bob (user) profile behaviour:")
    # Start of Selection
    if args.voice:
        raw = input(
            "Describe the user behaviour (or press 'r' to record voice): ",
        ).strip()
        if raw.lower() == "r":
            print("Recording user behaviour. Press Enter to stop.")
            audio = record_until_enter()
            # Transcribe the recording
            user_behaviour = (
                await transcribe_deepgram(audio)
            ).strip() or "You are a customer with a billing issue"
        else:
            user_behaviour = raw or "You are a customer with a billing issue"
    else:
        user_behaviour = (
            input(
                "Describe the user behaviour (e.g., 'You are a customer with a billing issue'): ",
            ).strip()
            or "You are a customer with a billing issue"
        )

    print("Configure Alice (assistant) profile behaviour:")
    if args.voice:
        raw = input(
            "Describe the assistant behaviour (or press 'r' to record voice): ",
        ).strip()
        if raw.lower() == "r":
            print("Recording assistant behaviour. Press Enter to stop.")
            audio = record_until_enter()
            assistant_behaviour = (
                await transcribe_deepgram(audio)
            ).strip() or "You are a utility provider support agent"
        else:
            assistant_behaviour = raw or "You are a utility provider support agent"
    else:
        assistant_behaviour = (
            input(
                "Describe the assistant behaviour (e.g., 'You are a utility provider support agent'): ",
            ).strip()
            or "You are a utility provider support agent"
        )

    alice = make_agent("alice", "Bob", "Alice", "+1001", "+1002", assistant_behaviour)
    bob = make_agent("bob", "Alice", "Bob", "+1002", "+1001", user_behaviour)

    last_message = None
    while True:
        print(COMMANDS_HELP)
        cmd = input("Command> ").strip().lower()

        if cmd in ("exit", "quit"):
            break
        if cmd in ("help", "h"):
            print(COMMANDS_HELP)
            continue
        if cmd not in ("start", "s", "continue", "c", "medium", "m"):
            print("Invalid command. Please try again.")
            continue

        if cmd in ("start", "s"):
            resp = await simulate_turn(
                bob,
                f"Start the {MEDIUM} conversation.",
                start=True,
            )
            last_message = resp
            print("Bob (user)>  ", last_message, "\n")
        elif cmd in {"medium", "m"}:
            new_medium = input("Select medium (phone, sms, email): ").strip().lower()
            if new_medium in ("phone", "sms", "email"):
                MEDIUM = new_medium
                print(f"Communication medium changed to: {MEDIUM}")
            else:
                print("Invalid medium. Please choose 'phone', 'sms', or 'email'.")
            last_message = None
            continue

        if last_message is None:
            print(
                f"No conversation started. Use 'start' first. Current medium: {MEDIUM}",
            )
            continue

        # run 3 exchange cycles
        for _ in range(3):
            # Alice turn
            alice_reply = await simulate_turn(alice, last_message)
            print("Alice (assistant)> ", alice_reply, "\n")
            if args.voice:
                speak(alice_reply)
            # Bob turn
            bob_reply = await simulate_turn(bob, alice_reply)
            print("Bob (user)>  ", bob_reply, "\n")
            last_message = bob_reply
    print("Simulation ended.")


if __name__ == "__main__":
    asyncio.run(simulate())
