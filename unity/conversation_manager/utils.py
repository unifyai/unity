import requests
from unify.utils import http

from unity.settings import SETTINGS


# admin headers and URLs
admin_headers = {
    "Authorization": f"Bearer {SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()}",
}
unity_comms_url = SETTINGS.conversation.COMMS_URL


# dispatch LiveKit agent
def dispatch_livekit_agent(livekit_agent_name: str, room_name: str = None):
    try:
        if not room_name:
            room_name = livekit_agent_name
        response = http.post(
            f"{unity_comms_url}/phone/dispatch-livekit-agent",
            headers=admin_headers,
            json={"livekit_agent_name": livekit_agent_name, "room_name": room_name},
            timeout=1,
        )
        if response.status_code != 200:
            print(f"Failed to dispatch LiveKit agent. {response.text}")
            return False
        else:
            print("LiveKit agent dispatched")
    except requests.exceptions.Timeout:
        print("LiveKit agent dispatched (timeout)")
    return True
