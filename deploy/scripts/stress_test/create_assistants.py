from dotenv import load_dotenv

load_dotenv()
import os
import asyncio
import aiohttp
import random
import string

# Configuration
API_KEY = os.getenv("UNIFY_KEY")
BASE_URL = os.getenv("ORCHESTRA_URL") + "/assistant"
NUM_ASSISTANTS = 10


async def create_assistant(session, index):
    """
    Creates a single assistant using the Orchestra API.
    """
    # Generate a random letter-only suffix to satisfy UNICODE_NAME_RE: ^[^\W\d_](?:[^\W\d_]|[ .'-])*$
    suffix = "".join(random.choices(string.ascii_letters, k=5))
    payload = {
        "first_name": "StressTest",
        "surname": f"Assistant{suffix}",
        "age": 25,
        "nationality": "US",
        "profile_photo": None,
        "profile_video": None,
        "about": "Stress testing assistant.",
        "voice_id": "ThT5KcBeYPX3keUQqHPh",
        "voice_provider": "elevenlabs",
        "voice_mode": "tts",
        "timezone": "Asia/Kolkata",
        "is_user_desktop": False,
        "desktop_mode": None,
        "max_parallel": 10,
        "weekly_limit": 40,
        "create_infra": True,
        "pre_hire_chat": [],
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        async with session.post(BASE_URL, json=payload, headers=headers) as response:
            status_code = response.status
            data = await response.json()
            if status_code == 200:
                print(
                    f"[{index}] Success: Created assistant {data.get('info', {}).get('agent_id')}",
                )
                return data
            else:
                print(
                    f"[{index}] Failed: {status_code} - {data.get('detail', 'No detail provided')}",
                )
                return None
    except Exception as e:
        print(f"[{index}] Error: {str(e)}")
        return None


async def main():
    print(f"Starting creation of {NUM_ASSISTANTS} assistants...")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(NUM_ASSISTANTS):
            print(f"[{i + 1}] Creating assistant...")
            result = await create_assistant(session, i + 1)
            if result is not None:
                print(
                    f"[{i + 1}] Success: Created assistant {result.get('info', {}).get('agent_id')}",
                )
            else:
                print(f"[{i + 1}] Failed: {result.get('detail', 'No detail provided')}")


if __name__ == "__main__":
    asyncio.run(main())
