from dataclasses import dataclass


@dataclass
class Assistant:
    job_name: str
    user_id: str
    assistant_id: str
    user_name: str
    assistant_name: str
    assistant_age: str
    assistant_region: str
    assistant_timezone: str
    assistant_about: str
    assistant_number: str
    assistant_email: str
    user_number: str
    user_email: str = None
    voice_provider: str = "cartesia"
    voice_id: str = None
    assistant_job_title: str = ""
