import requests
from unify.utils import http

from unity.settings import SETTINGS


# admin headers and URLs
admin_headers = {
    "Authorization": f"Bearer {SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()}"
}
unity_comms_url = SETTINGS.conversation.COMMS_URL


# dispatch agent
def dispatch_agent(agent_name: str, room_name: str = None):
    try:
        if not room_name:
            room_name = agent_name
        response = http.post(
            f"{unity_comms_url}/phone/dispatch-agent",
            headers=admin_headers,
            json={"agent_name": agent_name, "room_name": room_name},
            timeout=1,
        )
        if response.status_code != 200:
            print(f"Failed to dispatch agent. {response.text}")
            return False
        else:
            print("Agent dispatched")
    except requests.exceptions.Timeout:
        print("Agent dispatched (timeout)")
    return True
