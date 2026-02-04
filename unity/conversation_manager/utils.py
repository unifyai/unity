import requests

from unity.settings import SETTINGS

# admin headers and URLs
admin_headers = {
    "Authorization": f"Bearer {SETTINGS.ORCHESTRA_ADMIN_KEY.get_secret_value()}",
}
unity_comms_url = SETTINGS.conversation.COMMS_URL


# dispatch LiveKit agent
def dispatch_livekit_agent(livekit_agent_name: str, room_name: str = None):
    """
    Dispatch a LiveKit agent via the communication service.

    This is a fire-and-forget operation - we dispatch and move on regardless of
    the result. The function is resilient to:
    - Missing UNITY_COMMS_URL (common in local/test environments)
    - Network timeouts (expected behavior)
    - Connection errors (service unavailable)

    Returns True if dispatch was attempted, False if skipped due to missing config.
    """
    if not unity_comms_url:
        print(
            "[dispatch_livekit_agent] Skipping: UNITY_COMMS_URL not configured. "
            "Set this to enable LiveKit agent dispatch.",
        )
        return False

    try:
        if not room_name:
            room_name = livekit_agent_name
        # Fire-and-forget: use requests.post directly (not unify.utils.http)
        # to avoid retry logic. Timeout is expected; we dispatch and move on.
        response = requests.post(
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
        # Timeout is expected - the dispatch endpoint may be slow
        print("LiveKit agent dispatched (timeout)")
    except requests.exceptions.RequestException as e:
        # Connection errors, DNS failures, etc. - don't crash, just log
        print(f"[dispatch_livekit_agent] Request failed (non-fatal): {e}")
        return False
    return True
