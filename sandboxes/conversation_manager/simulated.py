import asyncio
import unify
from sandboxes.utils import build_cli_parser, activate_project
from unity.conversation_manager.prompt_builders import (
    build_call_sys_prompt,
    build_user_agent_prompt,
)
from unity.conversation_manager.events import PhoneUtteranceEvent


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
        "task_context": None,
        # "task_context": {
        #     "name": assistant_behaviour,
        #     "description": assistant_behaviour,
        # },
        "history": [],
        "client": client,
    }


async def simulate_turn(agent, message):
    # record user utterance via PhoneUtteranceEvent payload
    ue = PhoneUtteranceEvent(timestamp=None, content=message, role="User")
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
        call_purpose="general",
        past_events=agent["history"],
        inflight_events=agent["history"][-1:],
        tool_use_handles=None,
    )
    # run real AsyncUnify
    client = agent["client"]
    client.set_system_message(sys_prompt)
    # single-turn generate
    resp = await client.generate(user_message=ua_prompt)
    # record assistant response via PhoneUtteranceEvent payload
    reply = getattr(resp, "phone_utterance", None) or str(resp)
    ae = PhoneUtteranceEvent(timestamp=None, content=reply, role="Assistant")
    agent["history"].append(ae.to_dict())
    return reply


async def simulate():
    parser = build_cli_parser("Raw two-agent simulation")
    args = parser.parse_args()
    activate_project(args.project_name, args.overwrite)

    # Input for customised user profile
    print("Configure Bob (user) profile behaviour:")
    user_behaviour = (
        input(
            "Describe the user behaviour (e.g., 'You are a customer with a billing issue'): ",
        ).strip()
        or "You are a customer with a billing issue"
    )

    print("Configure Alice (assistant) profile behaviour:")
    assistant_behaviour = (
        input(
            "Describe the assistant behaviour (e.g., 'You are a utility provider support agent'): ",
        ).strip()
        or "You are a utility provider support agent"
    )

    alice = make_agent("alice", "Bob", "Alice", "+1001", "+1002", assistant_behaviour)
    bob = make_agent("bob", "Alice", "Bob", "+1002", "+1001", user_behaviour)

    print("Raw two-agent simulation. Commands: 'start', 'continue', 'exit'")
    last_message = None
    while True:
        cmd = input("Command> ").strip().lower()
        if cmd in ("exit", "quit"):
            break
        if cmd not in ("start", "continue"):
            print("Use 'start' to begin, 'continue' for next 5 turns, or 'exit'.")
            continue
        if cmd == "start":
            # Use unify AsyncClient and generate first response based on user_behaviour
            client = unify.AsyncUnify(endpoint="gpt-4.1@openai")
            resp = await client.generate(
                user_message=f"You are Bob, and you are calling Alice. Greet in one sentence with this purpose: {user_behaviour}",
            )
            last_message = getattr(resp, "phone_utterance", None) or str(resp)
            print("Bob (user)>  ", last_message, "\n")
        if last_message is None:
            print("No conversation started. Use 'start' first.")
            continue
        # run 5 exchange cycles
        for _ in range(3):
            # Alice turn
            alice_reply = await simulate_turn(alice, last_message)
            print("Alice (assistant)> ", alice_reply, "\n")
            # Bob turn
            bob_reply = await simulate_turn(bob, alice_reply)
            print("Bob (user)>  ", bob_reply, "\n")
            last_message = bob_reply
    print("Simulation ended.")


if __name__ == "__main__":
    asyncio.run(simulate())
