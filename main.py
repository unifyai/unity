import os

from dotenv import load_dotenv
from vapi_python.vapi_python import DailyCall, create_web_call

from sys_msgs import vocal_request_taker_sys_msg

load_dotenv()

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
        "model": "gpt-3.5-turbo",
        "voice": "jennifer-playht",
        "recordingEnabled": True,
        "interruptionsEnabled": False,
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
