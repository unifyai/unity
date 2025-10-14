import os
import requests
from unify.utils import http


# admin headers and URLs
admin_headers = {"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"}
unity_comms_url = os.getenv("UNITY_COMMS_URL")


# dispatch agent
def dispatch_agent(agent_name: str):
    try:
        response = http.post(
            f"{unity_comms_url}/phone/dispatch-agent",
            headers=admin_headers,
            json={"agent_name": agent_name},
            timeout=1,
            # raise_for_status=False,
        )
        if response.status_code != 200:
            print(f"Failed to dispatch agent. {response.text}")
            return False
        else:
            print("Agent dispatched")
    except requests.exceptions.Timeout:
        print("Agent dispatched (timeout)")
    return True
