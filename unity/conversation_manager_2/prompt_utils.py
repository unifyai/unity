from collections import deque


def add_spaces(string: str, num_spaces: int=4):
    ls = string.split("\n")
    return "\n".join(num_spaces * " "+ l for l in ls)

class NotificationBar:
    def __init__(self, notifs):
        self.notifs = notifs
    
    def __str__(self):
        return "\n".join([f"[Notification] {n}" for n in self.notifs])

class ThreadMessage:
    def __init__(self, name, content, timestamp):
        self.name = name
        self.content = content
        self.timestamp = timestamp

    def __str__(self):
        return f"""{self.name}@{self.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}: {self.content}"""

class ContactThread:
    def __init__(self, thread_name, max_len=15):
        self.thread_name = thread_name
        self.messages = deque(maxlen=max_len)
    
    def push_message(self, m):
        self.messages.append(m)
    
    def __bool__(self):
        return bool(self.messages)
    
    def __str__(self):
        thread_content = ""
        for m in self.messages:
            thread_content += str(m)
        thread_content = thread_content.strip()
        return f"""
<{self.thread_name}>
{add_spaces(thread_content)}
</{self.thread_name}>""".strip()

class ConversationContact:
    def __init__(self, name, is_boss=False, on_phone=False):
        self.name = name
        self.is_boss = is_boss
        self.on_phone = on_phone
        self.threads = {
            "sms": ContactThread("sms"),
            "email": ContactThread("email"),
            "phone": ContactThread("phone")
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
<contact name="{self.name}" is_boss="{self.is_boss}">
{add_spaces(threads_content)}
</contact>""".strip()