from dotenv import load_dotenv

load_dotenv()
import os
import asyncio
import json
import aiohttp

API_KEY = os.getenv("UNIFY_KEY")
BASE_URL = os.getenv("ORCHESTRA_URL") + "/assistant"
ASSISTANT_IDS = []


async def delete_assistant(session, assistant_id):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{BASE_URL}/{assistant_id}"
    try:
        async with session.delete(url, headers=headers) as response:
            text = await response.text()
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                print(f"  [{assistant_id}] Failed: {response.status} - {text[:500]}")
                return False
            if response.status == 200:
                print(f"  [{assistant_id}] Deleted")
                return True
            else:
                print(
                    f"  [{assistant_id}] Failed: {response.status} - {data.get('detail', text[:500])}",
                )
                return False
    except Exception as e:
        print(f"  [{assistant_id}] Error: {e}")
        return False


async def main():
    print(f"Deleting {len(ASSISTANT_IDS)} assistant(s)...")
    async with aiohttp.ClientSession() as session:
        for aid in ASSISTANT_IDS:
            await delete_assistant(session, aid)


if __name__ == "__main__":
    asyncio.run(main())
