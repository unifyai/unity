import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()


async def send_whatsapp_message(from_number: str, to_number: str, message: str) -> bool:
    """
    Send a WhatsApp message using the WhatsApp Business API.

    Args:
        from_number: The sender's phone number
        to_number: The recipient's phone number
        message: The message content to send

    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    try:
        print(f"Sending WhatsApp message from {from_number} to {to_number}: {message}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('UNITY_COMMS_URL')}/whatsapp/send-text",
                json={
                    "from": from_number,
                    "to": to_number,
                    "body": message,
                },
            ) as response:
                if response.status != 200:
                    print(f"Failed to send WhatsApp message. Status: {response.status}")
                    return False

                response_text = await response.text()
                print(f"Response: {response_text}")
                return True
    except aiohttp.ClientError as e:
        print(f"Network error while sending WhatsApp message: {e}")
        return False
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return False


async def send_sms(from_number: str, to_number: str, message: str) -> bool:
    """
    Send an SMS message using the SMS provider API.

    Args:
        from_number: The sender's phone number
        to_number: The recipient's phone number
        message: The message content to send

    Returns:
        bool: True if message was sent successfully, False otherwise
    """
    try:
        print(f"Sending SMS from {from_number} to {to_number}: {message}")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{os.getenv('UNITY_COMMS_URL')}/phone/send-text",
                json={
                    "From": from_number,
                    "To": to_number,
                    "Body": message,
                },
            ) as response:
                if response.status != 200:
                    print(f"Failed to send SMS. Status: {response.status}")
                    return False

                response_text = await response.text()
                print(f"Response: {response_text}")
                return True
    except aiohttp.ClientError as e:
        print(f"Network error while sending SMS: {e}")
        return False
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return False


async def handle_message_action(action_type: str, **kwargs) -> bool:
    """
    Handle different types of message actions based on the action type.

    Args:
        action_type: The type of action to perform (whatsapp, sms, telegram, email)
        **kwargs: Additional arguments needed for the specific action

    Returns:
        bool: True if action was successful, False otherwise
    """
    action_map = {"whatsapp": send_whatsapp_message, "sms": send_sms}

    if action_type not in action_map:
        print(f"Unknown action type: {action_type}")
        return False

    try:
        return await action_map[action_type](**kwargs)
    except Exception as e:
        print(f"Error handling message action {action_type}: {e}")
        return False
