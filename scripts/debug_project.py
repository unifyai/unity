import argparse
from dotenv import load_dotenv
load_dotenv()
import os
import requests
import unify

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", type=bool, default=False)
    args = parser.parse_args()
    unify.activate("Debug")
    if args.overwrite:
        unify.create_context("startup_events")
    users = [
        "7bea302d-2518-48ac-b21d-61a0c53c5d0f",
        # "clxlmko0900539b72enccyi1o",
        "clummoqze00002hdndizy7339",
        "clc0fxxko0009s601fsb1o5sz",
        "26b19a19-7cc0-467b-b369-636ad533cccc",
        "cli3t38uc0000s60k5zmgj8ez",
        "0951a71f-0c5a-4858-a980-e173b13c64c4",
        "67abcd12-1fac-4a8f-afe9-c54698c96971",
    ]
    for user in users:
        response = requests.post(
            f"{os.getenv('UNIFY_BASE_URL')}/admin/share-project",
            json={
                "from_user_id": "clxlmko0900539b72enccyi1o",
                "to_user_id": user,
                "project_name": "Debug",
            },
            headers={"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"},
        )
        print("Response sharing with", user, response.text)
