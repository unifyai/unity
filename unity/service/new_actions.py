import aiohttp
import os
from dotenv import load_dotenv
from unity.contact_manager.contact_manager import ContactManager
from unity.service.main import EventManager
from service.comms_agent import CommsAgent
from service.events import PhoneCallInitiatedEvent

load_dotenv()

headers = {"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"}


async def send_whatsapp_message(
    contact_id: int,
    content: str,
    *,
    cm: ContactManager,
) -> str:
    """
    Send a WhatsApp message using the WhatsApp Business API.

    Args:
        contact_id: The ID of the contact to send the message to
        content: The message content to send
        cm: The contact manager instance

    Returns:
        str: A string indicating the result of the action
    """
    from_number = os.getenv("ASSISTANT_NUMBER")
    contacts = cm._search_contacts(filter=f"contact_id == {contact_id}")
    if not contacts:
        print(f"Contact with ID {contact_id} not found")
        return "Message not sent: Contact not found"
    to_number = contacts[0].whatsapp_number

    try:
        print(
            f"Sending WhatsApp message from {from_number} to ID {to_number}: {content}",
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('UNITY_COMMS_URL')}/whatsapp/send-text",
                headers=headers,
                json={
                    "from": from_number,
                    "to": to_number,
                    "body": content,
                },
            ) as response:
                if response.status != 200:
                    print(f"Failed to send WhatsApp message. Status: {response.status}")
                    return "Message not sent: Failed to send WhatsApp message"

                response_text = await response.text()
                print(f"Response: {response_text}")
                return "Message sent successfully"
    except aiohttp.ClientError as e:
        print(f"Network error while sending WhatsApp message: {e}")
        return "Message not sent: Network error"
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return "Message not sent: Error"


async def send_sms(contact_id: int, content: str, *, cm: ContactManager) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        contact_id: The ID of the contact to send the message to
        content: The message content to send
        cm: The contact manager instance

    Returns:
        str: A string indicating the result of the action
    """
    from_number = os.getenv("ASSISTANT_NUMBER")
    contacts = cm._search_contacts(filter=f"contact_id == {contact_id}")
    if not contacts:
        print(f"Contact with ID {contact_id} not found")
        return "Message not sent: Contact not found"
    to_number = contacts[0].phone_number

    try:
        print(f"Sending SMS from {from_number} to ID {to_number}: {content}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('UNITY_COMMS_URL')}/phone/send-text",
                headers=headers,
                json={
                    "From": from_number,
                    "To": to_number,
                    "Body": content,
                },
            ) as response:
                if response.status != 200:
                    print(f"Failed to send SMS. Status: {response.status}")
                    return "Message not sent: Failed to send SMS"

                response_text = await response.text()
                print(f"Response: {response_text}")
                return "Message sent successfully"
    except aiohttp.ClientError as e:
        print(f"Network error while sending SMS: {e}")
        return "Message not sent: Network error"
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return "Message not sent: Error"


async def send_email(contact_id: int, content: str, *, cm: ContactManager) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        contact_id: The ID of the contact to send the email to
        content: The message content to send
        cm: The contact manager instance

    Returns:
        str: A string indicating the result of the action
    """
    from_email = os.getenv("ASSISTANT_EMAIL")  # todo: placeholder for env var key
    contacts = cm._search_contacts(filter=f"contact_id == {contact_id}")
    if not contacts:
        print(f"Contact with ID {contact_id} not found")
        return "Message not sent: Contact not found"
    to_email = contacts[0].email_address

    try:
        print(f"Sending email from {from_email} to {to_email}: {content}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('UNITY_COMMS_URL')}/email/send",
                headers=headers,
                json={
                    "from": from_email,
                    "to": to_email,
                    "body": content,
                },
            ) as response:
                if response.status != 200:
                    print(f"Failed to send email. Status: {response.status}")
                    return "Message not sent: Failed to send email"

                response_text = await response.text()
                print(f"Response: {response_text}")
                return "Message sent successfully"
    except aiohttp.ClientError as e:
        print(f"Network error while sending email: {e}")
        return "Message not sent: Network error"
    except Exception as e:
        print(f"Error sending email: {e}")
        return "Message not sent: Error"


# async def send_call(contact_id: int, *, cm: ContactManager) -> str:
#     """
#     Send a call using the call provider API.

#     Args:
#         contact_id: The ID of the contact to send the call to
#         cm: The contact manager instance
#     Returns:
#         str: A string indicating the result of the action
#     """
#     from_number = os.getenv("ASSISTANT_NUMBER")
#     contacts = cm._search_contacts(filter=f"contact_id == {contact_id}")
#     if not contacts:
#         print(f"Contact with ID {contact_id} not found")
#         return "Call not sent: Contact not found"
#     to_number = contacts[0].phone_number

#     try:
#         print(f"Sending call from {from_number} to ID {contact_id}")
#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 f"{os.getenv('UNITY_COMMS_URL')}/phone/send-call",
#                 headers=headers,
#                 json={"From": from_number, "To": to_number, "NewCall": "true"},
#             ) as response:
#                 if response.status != 200:
#                     print(f"Failed to send call. Status: {response.status}")
#                     return "Call not sent: Failed to send call"

#                 response_text = await response.text()
#                 print(f"Response: {response_text}")
#                 return "Call sent successfully"
#     except aiohttp.ClientError as e:
#         print(f"Network error while sending call: {e}")
#         return "Call not sent: Network error"
#     except Exception as e:
#         print(f"Error sending call: {e}")
#         return "Call not sent: Error"


async def make_call(
    contact_id: int,
    *,
    cm: ContactManager,
    agent: CommsAgent,
    em: EventManager,
) -> str:
    """
    Make a call using the call provider API.

    Args:
        contact_id: The ID of the contact to make the call to
        cm: The contact manager instance
        agent: The comms agent instance
        em: The event manager instance

    Returns:
        str: A string indicating the result of the action
    """
    from_number = os.getenv("ASSISTANT_NUMBER")
    contacts = cm._search_contacts(filter=f"contact_id == {contact_id}")
    if not contacts:
        print(f"Contact with ID {contact_id} not found")
        return "Call not made: Contact not found"
    to_number = contacts[0].phone_number

    # approach 1:publish the event to the event manager
    em.publish(
        {
            "topic": "user_agent",  # todo: check the actual topic
            "event": PhoneCallInitiatedEvent(
                contact_id=contact_id,
                from_number=from_number,
                to_number=to_number,
            ).to_dict(),
        },
    )

    # approach 2: publish through agent directly
    agent.send_call()  # handles everything, need to generalise

    agent.publish(
        {
            "topic": "user_agent",  # todo: check the actual topic
            "event": PhoneCallInitiatedEvent(
                contact_id=contact_id,
                from_number=from_number,
                to_number=to_number,
            ).to_dict(),
        },
    )
