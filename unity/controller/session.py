import os
from browserbase import Browserbase
from browserbase.types import SessionCreateResponse
from browserbase.types.sessions import LogListResponse, RecordingRetrieveResponse
from dotenv import load_dotenv

load_dotenv()


# initialize browserbase client
bb = Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"])


# session management
def create_session(
    with_context: bool = False, stealth_mode: bool = False
) -> SessionCreateResponse:
    session = bb.sessions.create(
        project_id=os.environ["BROWSERBASE_PROJECT_ID"],
        # keep_alive=True,
    )
    return session


def close_session(session_id: str) -> bool:
    session_res = bb.sessions.update(
        session_id,
        project_id=os.environ["BROWSERBASE_PROJECT_ID"],
        status="REQUEST_RELEASE",
    )
    return session_res.status in ("TIMED_OUT", "COMPLETED")


def get_logs(session_id: str) -> LogListResponse:
    return bb.sessions.logs.list(session_id)


# live view and session replay
def get_live_view_urls(session_id: str) -> list[str]:
    live_view_links = bb.sessions.debug(session_id)
    all_tabs = live_view_links.pages
    return [tab.debuggerFullscreenUrl for tab in all_tabs]


def get_recording(session_id: str) -> RecordingRetrieveResponse:
    return bb.sessions.recording.retrieve(session_id)
