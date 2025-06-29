# Learning from Video
import time
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


def reason(*args, output_format=None, **kwargs):
    # 0-shot LLM reasoning step (single LLM call)
    pass


def multi_step_reason(*args, output_format=None, **kwargs):
    # multi-step LLM reasoning via a tool loop (multiple LLM calls)
    pass


# Extract Video

browser = Browser()
browser.multi_step("Go to {video_url}")
browser.start_recording()
duration = (
    browser.observe(
        "How long is the video, rounded *up* in seconds?",
        output_format=int,
    )
    * 1.1
)  # with buffer
browser.act("press play")
time.sleep(duration)
browser.stop_recording()


def perform_task(past_failures: list[str]):

    # Task description

    def nearest(video, timestamp):
        # find nearest video frame to timestamp
        pass

    def extract_image(timestamp: datetime):
        return nearest(browser.video, timestamp)

    sys_msg = "given the following timestamped transcript, past failed task decomposition explanations, and a tool to extract full images at any timestamp, please extract an overall high-level description of the task that is being performed."
    task_description: str = multi_step_reason(
        sys_msg,
        past_failures,
        browser.transcript,
        task_steps,
        tools=extract_image,
    )

    # Parse and perform task steps

    class AnnotatedImage(BaseModel):
        image: str = Field(
            description="An image containing helpful guiding information for a single task step",
        )
        annotation: str = Field(
            description="A helpful annotation, to explain the context of the image in relation to the single task step.",
        )

    class TaskStep(BaseModel):
        description: str = Field(
            description="Clear, intuitive and fully repeatable description for this step in the overall task.",
        )
        images: list[AnnotatedImage] = Field(
            description="List of annotated images, which aid in explaining exactly how to perform this single step of the task.",
        )

    task_steps: dict[datetime, TaskStep] = dict()
    task_completed = False

    sys_msg = "given the following overall task description, timestamped transcript, preceeding task steps, and a tool to extract full images at any timestamp, please detect the {nth} step in the overall task, and return the timestamps at which this task is started and completed in the video."

    browser.multi_step("clear all tabs, but keep the browser open")

    start_ts = browser.video.start_ts
    while not task_completed:
        task_step, end_ts = multi_step_reason(
            sys_msg,
            start_ts,
            task_description,
            browser.transcript,
            task_steps,
            tools=extract_image,
        )

        browser_state = browser.state
        step_completed = False
        past_failures = list()
        while not step_completed:
            browser.seed(browser_state)
            browser.multi_step(task_step, past_failures)
            step_completed, error = multi_step_reason(
                f"has the following step been performed correctly?\n"
                "Task step to check:{task_step}\n"
                "True browser actions:{browser.actions}\n"
                "If not, was it due to an error in the task execution, or an error in the task description?",
            )
            if error.type == "exection":
                past_failures.append(error.explanation)
                continue
            elif error.type == "description":
                return False, error.explanation

        task_steps[start_ts] = task_step
        start_ts = end_ts
    return True, "task completed"


# run until the task completes, potentially re-parsing the task if there are errors:
task_completed = False
past_failures = list()
while not task_completed:
    task_completed, explanation = perform_task(past_failures)
    past_failures.append(explanation)

# Finally, take the full *correct* list of browser actions, and decompose into repeatable functions
