import asyncio
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
ADAPTER_URL = "https://unity-adapters-staging-ky4ja5fxna-uc.a.run.app"
ADMIN_KEY = os.getenv("ORCHESTRA_ADMIN_KEY")
ASSISTANT_IDS = list(range(587, 597))  # based on whatever are the assistant IDs
CONTACT_ID = 1  # the boss contact ID
SILLY_MESSAGES = [
    "Knock knock! Who's there? A stress test!",
    "If you're reading this, you've been successfully stress-tested.",
    "Beep boop, I am a robot sending you a silly message.",
    "How many assistants does it take to change a lightbulb? 10, apparently.",
    "This is a message from the future. It says 'Hi'.",
    "Do you dream of electric sheep, or just stress tests?",
    "I'm not saying I'm a stress test, but have you ever seen me and a stress test in the same room?",
    "Error 404: SILLY_MESSAGE not found. Just kidding.",
    "Why did the assistant cross the road? To get to the other side of the stress test.",
    "Keep calm and carry on stress testing.",
]


async def send_unify_message(session, assistant_id, message):
    url = f"{ADAPTER_URL}/unify/message"
    payload = {
        "assistant_id": str(assistant_id),
        "contact_id": CONTACT_ID,
        "body": message,
    }
    headers = {
        "Authorization": f"Bearer {ADMIN_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            status = response.status
            text = await response.text()
            if status == 200:
                print(f"[Assistant {assistant_id}] Message sent successfully")
            else:
                print(
                    f"[Assistant {assistant_id}] Failed to send message: {status} - {text}",
                )
    except Exception as e:
        print(f"[Assistant {assistant_id}] Error sending message: {e}")


async def main():
    if not ADMIN_KEY:
        print("Error: ORCHESTRA_ADMIN_KEY is not set in environment.")
        return

    print(f"Sending parallel Unify messages to assistants {ASSISTANT_IDS}...")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, assistant_id in enumerate(ASSISTANT_IDS):
            message = SILLY_MESSAGES[i % len(SILLY_MESSAGES)]
            tasks.append(send_unify_message(session, assistant_id, message))

        await asyncio.gather(*tasks)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
