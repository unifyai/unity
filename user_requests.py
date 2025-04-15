import logging
import os
import random
import threading
import time

import requests
import unify
from dotenv import load_dotenv
from tqdm import tqdm
from vapi_python.vapi_python import DailyCall, create_web_call

from sys_msgs import vocal_request_taker_sys_msg

load_dotenv()
VAPI_PRIVATE_KEY = os.environ["VAPI_PRIVATE_KEY"]


def make_request():
    first_name = os.environ["FIRST_NAME"]
    num_conversations = 0
    summary = ""
    past_summary = f"A summary of the past f{num_conversations} conversations with f{first_name} are as follows:{summary}"
    assistant = {
        "assistant": {
            "firstMessage": f"Hey {first_name}, how can I help?",
            "context": vocal_request_taker_sys_msg.replace(
                "{first_name}",
                first_name,
            ),
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
    thread = threading.Thread(target=upload_call_logs, args=(call_id,))
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
    logging.info(f"uploading call logs for call id {call_id}...")
    failures = 0
    failure_limit = 10
    sleep_time = 30
    with tqdm(total=failure_limit, desc="Uploading logs from call") as pbar:
        while True:
            response = get_call_logs(call_id)
            if response.status_code != 200:
                if failures < failure_limit:
                    time.sleep(sleep_time + random.uniform(-2, 2))
                    failures += 1
                    pbar.update(1)
                    continue
                else:
                    raise Exception(f"Failed to get call logs for call {call_id}")
            response = response.json()
            if response["status"] == "in-progress":
                continue
            elif response["status"] == "ended":
                unify.log(context="Calls", **response)
                break
    logging.info(f"logs for call id {call_id} uploaded!")
