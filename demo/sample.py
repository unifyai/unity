# import asyncio
# from actions import send_sms

# async def main():
#     # Call send_sms with the specified parameters
#     result = await send_sms(
#         from_number="+17343611691",
#         to_number="6605382869",
#         # to_number="+919823987293",
#         message="hello world"
#     )

#     if result:
#         print("SMS sent successfully!")
#     else:
#         print("Failed to send SMS.")

# if __name__ == "__main__":
#     asyncio.run(main())


# from google.cloud import run_v2

# print(run_v2)


# from dotenv import load_dotenv

# load_dotenv()


# import os
# from twilio.rest import Client as TwilioClient


# account_sid = os.getenv("TWILIO_ACCOUNT_SID")
# auth_token = os.getenv("TWILIO_AUTH_TOKEN")
# twilio_client = TwilioClient(account_sid, auth_token)
# senders = twilio_client.messaging.v2.channels_senders.list("whatsapp")
# for sender in senders:
#     print(sender.properties)
#     print(sender.sid)
#     print(sender.configuration)
#     print(sender.profile)
#     print(sender.sender_id)
#     print(sender.url)
#     print(sender.webhook)
#     print()

# connect --room=Unity_###


# import json
# import os
# from google.cloud import pubsub_v1


# commit_sha = "297d4a3"  # "11bb61a"
# repo_pattern = "us-central1-docker.pkg.dev/responsive-city-458413-a2/unity/unity"
# project_id = "responsive-city-458413-a2"
# # "us-central1-docker.pkg.dev/responsive-city-458413-a2/unity/unity:11bb61a"

# # Initialize the publisher client
# publisher = pubsub_v1.PublisherClient()
# topic_path = publisher.topic_path(project_id, "unity-image-built")

# # Create the message payload with our minimal format
# message_data = {"_COMMIT_SHA": commit_sha, "_REPO_PATTERN": repo_pattern}

# # Convert to JSON string and encode as bytes
# message_json = json.dumps(message_data)
# message_bytes = message_json.encode("utf-8")

# print(f"Publishing message to {topic_path}")
# print(f"Message: {message_json}")

# try:
#     # Publish the message
#     future = publisher.publish(topic_path, message_bytes)
#     message_id = future.result()  # Wait for the publish to complete

#     print(f"✅ Message published successfully!")
#     print(f"Message ID: {message_id}")

# except Exception as e:
#     print(f"❌ Failed to publish message: {e}")


# import os
# from dotenv import load_dotenv

# from livekit import agents
# from livekit.agents import AgentSession, Agent, RoomInputOptions
# from livekit.plugins import (
#     openai,
#     noise_cancellation,
# )

# load_dotenv()


# class Assistant(Agent):
#     def __init__(self) -> None:
#         super().__init__(instructions="You are a helpful voice AI assistant.")


# async def entrypoint(ctx: agents.JobContext):
#     session = AgentSession(
#         llm=openai.realtime.RealtimeModel(
#             voice="coral"
#         )
#     )
#     print(f"Job metadata: {ctx.job.metadata}")
#     print(f"Room: {ctx.room}")
#     print(f"Attempting to connect to room: {ctx.room.name}")

#     try:
#         await ctx.connect()
#         print(f"✅ Successfully connected to room: {ctx.room.name}")
#         print(f"Room connection state: {ctx.room.connection_state}")
#     except Exception as e:
#         print(f"❌ Failed to connect to room: {e}")
#         print(f"Room connection state: {ctx.room.connection_state}")
#         raise

#     try:
#         await session.start(
#             room=ctx.room,
#             agent=Assistant(),
#             room_input_options=RoomInputOptions(
#                 # LiveKit Cloud enhanced noise cancellation
#                 # - If self-hosting, omit this parameter
#                 # - For telephony applications, use `BVCTelephony` for best results
#                 noise_cancellation=noise_cancellation.BVC(),
#             ),
#         )
#         print("✅ Agent session started successfully")
#     except Exception as e:
#         print(f"❌ Failed to start agent session: {e}")
#         raise

#     try:
#         await session.generate_reply(
#             instructions="Greet the user and offer your assistance."
#         )
#         print("✅ Generated initial greeting")
#     except Exception as e:
#         print(f"❌ Failed to generate initial greeting: {e}")
#         raise


# if __name__ == "__main__":
#     agent_name = f"unity_{os.environ['ASSISTANT_NUMBER']}"
#     print(f"Starting worker with agent name: {agent_name}")
#     agents.cli.run_app(agents.WorkerOptions(
#         entrypoint_fnc=entrypoint,
#         agent_name=agent_name  # This enables explicit dispatch mode
#     ))


"""
sip_dispatch_rule_id: "SDR_sfsMMaV6oCWE"
rule {
  dispatch_rule_individual {
    room_prefix: "+1"
  }
}
name: "Twilio Dispatch Rule"
"""

# from google.cloud import pubsub_v1

# subscriber = pubsub_v1.SubscriberClient()

# def handle_message(message):
#     print(f"Received message: {message}")

# pull_future = subscriber.subscribe(
#     "projects/responsive-city-458413-a2/subscriptions/unity-default-assistant-sub",
#     callback=handle_message
# )
# print(f"Subscribed to {pull_future}")

# try:
#     # Keep the main thread alive
#     pull_future.result()  # This will block indefinitely
# except KeyboardInterrupt:
#     pull_future.cancel()
#     print("Subscriber stopped.")


# from google.cloud import run_v2
# from google.iam.v1 import iam_policy_pb2, policy_pb2

# services_client = run_v2.ServicesClient()

# # Define the project and region
# project_id = "responsive-city-458413-a2"
# region = "us-central1"
# parent = f"projects/{project_id}/locations/{region}"

# print(f"Listing Cloud Run services in {parent}")

# # First, let's see what services actually exist
# try:
#     request = run_v2.ListServicesRequest(parent=parent)
#     services = services_client.list_services(request=request)

#     print("📋 Available Cloud Run services:")
#     for service in services:
#         service_name = service.name.split('/')[-1]  # Extract just the service name
#         print(f"  - {service_name}")
#         print(f"    Full path: {service.name}")
#         print(f"    URI: {service.uri}")
#         print()

#     if not any(services):
#         print("❌ No Cloud Run services found in this region")

# except Exception as e:
#     print(f"❌ Failed to list services: {e}")

# # Now let's try to update IAM for the correct service name
# # (You can update this after seeing what services exist)
# service_name = "unity-temp123"  # Updated to use existing service
# resource_path = f"projects/{project_id}/locations/{region}/services/{service_name}"

# print(f"\n🔧 Attempting to update IAM policy for: {resource_path}")

# # Use the correct IAM approach for Cloud Run v2
# try:
#     # Get current IAM policy using the working request approach
#     from google.iam.v1 import iam_policy_pb2
#     request = iam_policy_pb2.GetIamPolicyRequest(resource=resource_path)
#     policy = services_client.get_iam_policy(request=request)
#     print("✅ Successfully retrieved current IAM policy")
# except Exception as e:
#     print(f"❌ Failed to get IAM policy: {e}")
#     print("This likely means the service doesn't exist. Check the service list above.")
#     exit(1)

# # Check if allUsers already has Cloud Run Invoker role
# invoker_binding = None
# for binding in policy.bindings:
#     if binding.role == "roles/run.invoker":
#         invoker_binding = binding
#         break

# # Add allUsers to Cloud Run Invoker role
# from google.iam.v1 import policy_pb2

# if not invoker_binding:
#     invoker_binding = policy_pb2.Binding(role="roles/run.invoker", members=["allUsers"])
#     policy.bindings.append(invoker_binding)
#     print("➕ Added new binding for roles/run.invoker")
# elif "allUsers" not in invoker_binding.members:
#     invoker_binding.members.append("allUsers")
#     print("➕ Added allUsers to existing roles/run.invoker binding")
# else:
#     print("✅ allUsers already has roles/run.invoker access")

# # Set the updated policy using the correct request approach
# try:
#     request = iam_policy_pb2.SetIamPolicyRequest(resource=resource_path, policy=policy)
#     services_client.set_iam_policy(request=request)
#     print("✅ Successfully updated IAM policy - service now allows unauthenticated access")
# except Exception as e:
#     print(f"❌ Failed to set IAM policy: {e}")


# import requests


# response = requests.post(
#     "https://unity-comms-app-262420637606.us-central1.run.app/infra/service/create",
#     data={
#         "assistant_id": "default-assistant-2",
#         "user_name": "Ved",
#         "assistant_number": "+18507877970",
#         "user_number": "+919823987293",
#     },
# )
# print(response.json())


import asyncio
from dotenv import load_dotenv
import unify
from unity.contact_manager.contact_manager import ContactManager


load_dotenv()
unify.activate("ContactManagerIntegration")
# unify.set_trace_context("Traces")
# unify.set_context("Contacts")


async def main():
    contact_manager = ContactManager()
    handle = await contact_manager.ask(
        "Who do you think is the fastest superhero in the contact list?",
        _return_reasoning_steps=True,
    )

    print("handle", handle)
    while not handle.done():
        print("waiting for handle to be done")
        await asyncio.sleep(1)

    answer, _steps = await handle.result()
    print(answer)
    print(_steps)


if __name__ == "__main__":
    asyncio.run(main())
