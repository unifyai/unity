import aiohttp
import os
from dotenv import load_dotenv
from service.events import PhoneCallInitiatedCustomEvent
from service.utils import publish_event, get_contact_details

load_dotenv()

headers = {"Authorization": f"Bearer {os.getenv('ORCHESTRA_ADMIN_KEY')}"}


async def send_whatsapp_message(
    contact_id: int,
    content: str,
) -> str:
    """
    Send a WhatsApp message using the WhatsApp Business API.

    Args:
        contact_id: The ID of the contact to send the message to
        content: The message content to send

    Returns:
        str: A string indicating the result of the action
    """
    from_number = os.getenv("ASSISTANT_NUMBER")
    contacts = get_contact_details(contact_id)
    if not contacts:
        print(f"Contact with ID {contact_id} not found")
        return "Message not sent: Contact not found"
    to_number = contacts["whatsapp_number"]

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


async def send_sms_message(contact_id: int, content: str) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        contact_id: The ID of the contact to send the message to
        content: The message content to send

    Returns:
        str: A string indicating the result of the action
    """
    from_number = os.getenv("ASSISTANT_NUMBER")
    contacts = get_contact_details(contact_id)
    if not contacts:
        print(f"Contact with ID {contact_id} not found")
        return "Message not sent: Contact not found"
    to_number = contacts["phone_number"]

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


async def send_email(contact_id: int, content: str) -> str:
    """
    Send an SMS message using the SMS provider API.

    Args:
        contact_id: The ID of the contact to send the email to
        content: The message content to send

    Returns:
        str: A string indicating the result of the action
    """
    from_email = os.getenv("ASSISTANT_EMAIL")
    if not from_email:
        from_email = "unity.agent@unify.ai"  # todo: temp placeholder
        # print("No email address found for assistant")
        # return "Message not sent: No email address found for assistant"
    contacts = get_contact_details(contact_id)
    if not contacts:
        print(f"Contact with ID {contact_id} not found")
        return "Message not sent: Contact not found"
    to_email = contacts["email_address"]

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


async def make_call(contact_id: int, purpose: str) -> str:
    """
    Send a call using the call provider API.

    Args:
        contact_id: The ID of the contact to send the call to
        purpose: The purpose of the call
    Returns:
        str: A string indicating the result of the action
    """
    from_number = os.getenv("ASSISTANT_NUMBER")
    contacts = get_contact_details(contact_id)
    if not contacts:
        print(f"Contact with ID {contact_id} not found")
        return "Call not sent: Contact not found"
    to_number = contacts["phone_number"]

    # publish the event to the event manager
    publish_event(
        {
            "topic": to_number,
            "event": PhoneCallInitiatedCustomEvent(
                contact_id=contact_id,
                purpose=purpose,
            ).to_dict(),
        },
    )

    try:
        print(f"Sending call from {from_number} to {to_number}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('UNITY_COMMS_URL')}/phone/send-call",
                headers=headers,
                json={"From": from_number, "To": to_number, "NewCall": "true"},
            ) as response:
                if response.status != 200:
                    print(f"Failed to send call. Status: {response.status}")
                    return "Call not sent: Failed to send call"

                response_text = await response.text()
                print(f"Response: {response_text}")
                return "Call sent successfully"
    except aiohttp.ClientError as e:
        print(f"Network error while sending call: {e}")
        return "Call not sent: Network error"
    except Exception as e:
        print(f"Error sending call: {e}")
        return "Call not sent: Error"
