import json
import os
import random
import threading
import time
from typing import List

import requests
import unify
from dotenv import load_dotenv
from pydantic import BaseModel
from vapi_python.vapi_python import DailyCall, create_web_call

from sys_msgs import (
    extract_request_from_call_sys_msg,
    vocal_request_taker_sys_msg,
)

load_dotenv()
VAPI_PRIVATE_KEY = os.environ["VAPI_PRIVATE_KEY"]
FIRST_NAME = os.environ["FIRST_NAME"]


@unify.cached
def make_call():
    assistant = {
        "assistant": {
            "firstMessage": f"Hey {FIRST_NAME}, how can I help?",
            "context": vocal_request_taker_sys_msg,
            "model": "gpt-4o",
            "voice": "jennifer-playht",
            "recordingEnabled": True,
            "interruptionsEnabled": True,
        },
        "assistantOverrides": None,
    }
    call_id, web_call_url = create_web_call(
        api_url="https://api.vapi.ai",
        api_key="31071044-4dc6-454d-8ed9-c6a1851e726a",
        assistant=assistant,
    )
    client = DailyCall()
    client.join(web_call_url)
    input("press enter to end the call\n")
    client.leave()
    time.sleep(10)
    return call_id


def make_request():
    call_id = make_call()
    thread = threading.Thread(target=upload_request, args=(call_id,))
    thread.start()
    time.sleep(1)


def get_call_logs(call_id):
    return requests.get(
        f"https://api.vapi.ai/call/{call_id}",
        headers={
            "Authorization": f"Bearer {VAPI_PRIVATE_KEY}",
        },
    )


def upload_call_logs(call_id):
    print(f"uploading call logs for call id {call_id}...")
    failures = 0
    failure_limit = 10
    sleep_time = 30
    while True:
        response = get_call_logs(call_id)
        if response.status_code != 200:
            if failures < failure_limit:
                time.sleep(sleep_time + random.uniform(-2, 2))
                failures += 1
                continue
            else:
                raise Exception(f"Failed to get call logs for call {call_id}")
        response = response.json()
        if response["status"] == "in-progress":
            continue
        elif response["status"] == "ended":
            with unify.Context("Calls", overwrite=True):
                unify.log(
                    **{"call_id" if k == "id" else k: v for k, v in response.items()},
                )
            break
    print(f"logs for call id {call_id} uploaded!")
    return response


class Requests(BaseModel):
    list_of_requests: List[str]


def upload_request(call_id):
    response = upload_call_logs(call_id)
    client = unify.Unify("o3-mini@openai")
    client.set_system_message(extract_request_from_call_sys_msg)
    client.set_response_format(Requests)
    requests = client.generate(
        json.dumps(
            {k: v for k, v in response.items() if k in ("summary", "transcript")},
            indent=4,
        ),
    )
    requests = [{"request": v} for v in json.loads(requests)["list_of_requests"]]
    unify.create_logs(
        context="Requests",
        entries=requests,
    )
    print("uploaded requests")
