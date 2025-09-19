import base64
import re
import traceback


def _strip_quoted_text(text: str) -> str:
    """Remove quoted text and signatures from email content."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(">"):
            continue
        if re.match(r"On .+wrote:", stripped) or stripped.startswith(
            "-----Original Message-----"
        ):
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _header(headers, name: str) -> str:
    """Extract a specific header from email headers."""
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def _payload_text(payload) -> str:
    """Extract text content from email payload."""
    mime_type = payload.get("mimeType", "")
    if mime_type.startswith("text/") and payload.get("body", {}).get("data"):
        data = payload["body"]["data"]
        decoded = base64.urlsafe_b64decode(data.encode("utf-8"))
        latest = _strip_quoted_text(decoded.decode("utf-8", errors="replace"))
        return latest

    for part in payload.get("parts", []):
        txt = _payload_text(part)
        if txt:
            return txt
    return ""


def _gmail_thread_to_conversation(thread):
    """Convert a Gmail thread to a structured conversation."""
    convo = []
    for msg in thread.get("messages", []):
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        convo.append(
            {
                "sender": _header(headers, "From"),
                "to": (
                    [_addr.strip() for _addr in _header(headers, "To").split(",")]
                    if _header(headers, "To")
                    else []
                ),
                "cc": (
                    [_addr.strip() for _addr in _header(headers, "Cc").split(",")]
                    if _header(headers, "Cc")
                    else []
                ),
                "bcc": (
                    [_addr.strip() for _addr in _header(headers, "Bcc").split(",")]
                    if _header(headers, "Bcc")
                    else []
                ),
                "subject": _header(headers, "Subject"),
                "content": _payload_text(payload),
            }
        )
    return convo


def get_thread_id(user_id, history_id, gmail_service):
    """Process Gmail history and thread to extract conversation data."""
    try:
        # Get history events for label changes
        histories = (
            gmail_service.users()
            .history()
            .list(
                userId=user_id,
                startHistoryId=history_id,
                # historyTypes=["messageAdded", "labelAdded"],
            )
            .execute()
        )
        print(f"pre-histories: {histories}")

        # Safeguard for thread replies
        if "history" not in histories or not histories["history"]:
            histories["history"] = [
                (
                    gmail_service.users()
                    .messages()
                    .list(
                        userId=user_id,
                        q="is:unread newer_than:1d",
                    )
                    .execute()
                )
            ]

        # Process each history entry
        print(f"histories: {histories}")
        for history in histories["history"]:
            print(f"history: {history}")
            messages = history.get("messages", [])
            print(f"messages: {messages}")
            if len(messages) == 0:
                continue

            # Get the message details
            msg_id = messages[-1]["id"]
            message = (
                gmail_service.users()
                .messages()
                .get(userId=user_id, id=msg_id)
                .execute()
            )
            print(f"message: {message} {msg_id}")
            message_headers = message["payload"].get("headers", [])
            print(f"message_headers: {message_headers}")
            message_id_headers = [
                header
                for header in message_headers
                if header.get("name") == "Message-ID"
            ]
            if not message_id_headers:
                print(f"No Message-ID header found for message {msg_id}")
                continue
            message_id_header = message_id_headers[0]
            message_id = message_id_header.get("value")
            print(f"message_id: {message_id}")

            labels = message.get("labelIds", [])
            print(f"labels: {labels}")
            if labels and "UNREAD" not in labels:
                print(f"Message {msg_id} is read, skipping")
                continue

            gmail_service.users().messages().modify(
                userId=user_id, id=msg_id, body={"removeLabelIds": ["UNREAD"]}
            ).execute()

            # Get the thread for this message
            thread_id = message["threadId"]
            thread = (
                gmail_service.users()
                .threads()
                .get(userId=user_id, id=thread_id, format="full")
                .execute()
            )
            print(f"thread: {thread} {thread_id}")

            # Convert to conversation format
            conversation = _gmail_thread_to_conversation(thread)
            print(f"conversation: {conversation}")
            last_message = conversation[-1]
            print(f"last_message: {last_message}")

            # Return the conversation (or process it further as needed)
            return thread_id, message_id, last_message

        return None, None, None

    except Exception as e:
        print(f"Error processing history for user {user_id}: {str(e)}")
        traceback.print_exc()
        return None, None, None
