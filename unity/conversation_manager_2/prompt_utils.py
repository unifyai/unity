from collections import deque


def add_spaces(string: str, num_spaces: int = 4):
    ls = string.split("\n")
    return "\n".join(num_spaces * " " + l for l in ls)


class NotificationBar:
    def __init__(self):
        self.notifs = []

    def push_notif(self, type, n, timestamp=None):
        if timestamp:
            str_timestamp = timestamp.strftime("%A, %B %d, %Y at %I:%M %p")

        self.notifs.append(
            {
                "type": type,
                "content": n,
                "str_timestamp": str_timestamp,
                "timestamp": timestamp,
            },
        )

    def clear(self, timestamp=None):
        if timestamp:
            # print("comparing to", timestamp)
            self.notifs = [n for n in self.notifs if n["timestamp"] > timestamp]
        else:
            self.notifs = []

    def __str__(self):
        return "\n".join(
            [
                f"[{n['type'].title()} Notification @ {n['str_timestamp']}] {n['content']}"
                for n in self.notifs
            ],
        )


class ThreadMessage:
    def __init__(self, name, content, timestamp):
        self.name = name
        self.content = content
        self.timestamp = timestamp

    def __str__(self):
        return f"""[{self.name} @ {self.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}]: {self.content}"""


class EmailThreadMessage:
    def __init__(self, name, subject, body, timestamp):
        self.name = name
        self.subject = subject
        self.body = body
        self.timestamp = timestamp

    def __str__(self):
        return f"""[{self.name} @ {self.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}]:
Subject: {self.subject}
Body:
{self.body}
"""


class ContactThread:
    def __init__(self, thread_name, max_len=15):
        self.thread_name = thread_name
        self.messages = deque(maxlen=max_len)

    def push_message(self, m):
        self.messages.append(m)

    def __bool__(self):
        return bool(self.messages)

    def __str__(self):
        thread_content = "\n".join(str(m) for m in self.messages)
        thread_content = thread_content.strip()
        return f"""
<{self.thread_name}>
{add_spaces(thread_content)}
</{self.thread_name}>""".strip()


class ConversationContact:
    def __init__(
        self, id, name, is_boss=False, number=None, email=None, on_phone=False
    ):
        self.id = id
        self.name = name
        self.is_boss = is_boss
        self.on_phone = on_phone
        self.number = number
        self.email = email
        self.threads = {
            "sms": ContactThread("sms"),
            "email": ContactThread("email"),
            "phone": ContactThread("phone"),
        }

    def push_message(self, thread_name, message):
        self.threads[thread_name].push_message(message)

    def __str__(self):
        threads = []
        for t in self.threads.values():
            if t:
                threads.append(t)
        threads_content = "\n\n".join(str(t) for t in threads)
        return f"""
<contact id="{self.id}" name="{self.name}" is_boss="{self.is_boss}" phone_number="{self.number or ""}" email="{self.email or ""}">
{add_spaces(threads_content)}
</contact>""".strip()
