# Learning from Video
from enum import StrEnum
from datetime import datetime
from pydantic import BaseModel, Field


class Action(BaseModel):
    image_before: str = Field(
        description="Base64 screenshot immediately *before* the action was taken",
    )
    action: StrEnum = Field(description="The specific browser action that was taken")
    image_after: str = Field(
        description="Base64 screenshot immediately *after* the action was taken",
    )


class Browser:

    def act(self):
        # single step action, such as button click, typing text etc.
        pass

    def observe(self):
        # single step observation, answering any questions about the current browser state + screenshot
        pass

    def multi_step(self):
        # Uses Controller with `act` and `observe` as tools, can do more complex stuff like: "search google for 'dogs'" -> click search, type 'dogs', press enter etc.
        pass

    def start_recording(
        self,
        include_video: bool = True,
        include_transcript: bool = True,
    ):
        # starts recording the screen at high-frame rate
        pass

    def stop_recording(self):
        # stops recording the screen
        pass

    def seed(self):
        # reset the browser to this state
        pass

    @property
    def video(self) -> dict[datetime, str]:
        # datetime in UTC mapped to Base64 images
        pass

    @property
    def transcript(self) -> dict[datetime, str]:
        # datetime in UTC mapped to transcript chunks
        pass

    @property
    def actions(self) -> dict[datetime, Action]:
        # the full history of actions taken by the browser
        pass


class GoogleMeet(Browser):
    pass


# Start call
call = GoogleMeet(
    url="{google_meet_url}",
    context="You will be following user instructions to complete a task",
)
call.start_recording()

browser = Browser()

task_complete = False
while not task_complete:
    action = call.observe("What should we do next?")
    browser.multi_step(action)
    task_complete = call.observe("Is the task complete?")
