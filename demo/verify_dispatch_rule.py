import os
import asyncio
from dotenv import load_dotenv
from livekit import api

load_dotenv()


async def verify_dispatch_setup():
    """Verify the dispatch rule setup matches LiveKit documentation"""

    # Get LiveKit API client
    url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not url or not api_key or not api_secret:
        print("❌ Missing LiveKit credentials in environment")
        return

    livekit_api = api.LiveKitAPI(url=url, api_key=api_key, api_secret=api_secret)

    try:
        # list rooms
        request = api.ListRoomsRequest()
        response = await livekit_api.room.list_rooms(request)
        rooms = response.rooms
        print(len(rooms), rooms)

        request = api.ListSIPDispatchRuleRequest()
        response = await livekit_api.sip.list_sip_dispatch_rule(request)
        dispatch_rules = response.items
        for rule in dispatch_rules:
            print(rule)
            print()

        # request = api.DeleteSIPDispatchRuleRequest(sip_dispatch_rule_id="SDR_TEKHda68Jk8S")
        # response = await livekit_api.sip.delete_sip_dispatch_rule(request)
        # print(response)

        # general individual dispatch rule (that has always been there)
        # rule = api.SIPDispatchRule(
        #     dispatch_rule_individual = api.SIPDispatchRuleIndividual(
        #         room_prefix = 'unity-',
        #     )
        # )
        # request = api.CreateSIPDispatchRuleRequest(rule=rule, name="Twilio Dispatch Rule")
        # response = await livekit_api.sip.create_sip_dispatch_rule(request)
        # print(response)

        # rule = api.SIPDispatchRule(
        #     dispatch_rule_callee = api.SIPDispatchRuleCallee(
        #         room_prefix="unity-",
        #         randomize=False,
        #     )
        # )
        # request = api.CreateSIPDispatchRuleRequest(rule=rule)
        # response = await livekit_api.sip.create_sip_dispatch_rule(request)
        # print(response)

    except Exception as e:
        print(f"❌ Error verifying setup: {e}")
    finally:
        await livekit_api.aclose()


if __name__ == "__main__":
    asyncio.run(verify_dispatch_setup())
