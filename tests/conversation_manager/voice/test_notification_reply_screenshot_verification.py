"""
tests/conversation_manager/voice/test_notification_reply_screenshot_verification.py
=====================================================================================

Eval tests for the NotificationReplyEvaluator's ability to cross-reference
screenshots against ``Computer action executed:`` notifications.

Fast-path computer actions are blind single-shot attempts that frequently
fail silently (e.g., F11 doesn't fullscreen the browser). The evaluator
receives the assistant's screen screenshot alongside the notification.
These tests verify that the evaluator does NOT parrot a false completion
claim when the screenshot contradicts it.
"""

import base64
import io

import pytest
from PIL import Image, ImageDraw, ImageFont

from unity.common.llm_client import new_llm_client
from unity.conversation_manager.domains.notification_reply import (
    NotificationReplyEvaluator,
)
from unity.settings import SETTINGS

pytestmark = pytest.mark.eval

MODEL = SETTINGS.conversation.FAST_BRAIN_MODEL


def _make_windowed_browser_screenshot() -> str:
    """Synthesise a small screenshot of a clearly non-fullscreen browser window.

    The image shows a desktop background with a small browser window in the
    centre — complete with a visible title bar and surrounding desktop space,
    making it unambiguous that the browser is *not* fullscreen.
    """
    width, height = 640, 480
    img = Image.new("RGB", (width, height), color=(50, 50, 120))

    draw = ImageDraw.Draw(img)

    # Desktop area visible on all sides → clearly not fullscreen.
    win_left, win_top = 80, 60
    win_right, win_bottom = 560, 420

    # Title bar
    draw.rectangle(
        [win_left, win_top, win_right, win_top + 30],
        fill=(60, 60, 60),
    )
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text(
        (win_left + 10, win_top + 7),
        "Google - Chromium",
        fill=(220, 220, 220),
        font=font,
    )

    # Window control buttons (minimize / maximize / close)
    for i, colour in enumerate([(80, 80, 80), (80, 80, 80), (200, 60, 60)]):
        cx = win_right - 20 - i * 25
        draw.rectangle([cx - 8, win_top + 7, cx + 8, win_top + 23], fill=colour)

    # Browser content area (white)
    draw.rectangle(
        [win_left, win_top + 30, win_right, win_bottom],
        fill=(255, 255, 255),
    )

    # Address bar
    draw.rectangle(
        [win_left + 10, win_top + 40, win_right - 10, win_top + 60],
        fill=(240, 240, 240),
    )
    draw.text(
        (win_left + 20, win_top + 43),
        "google.com",
        fill=(50, 50, 50),
        font=font,
    )

    # Google-ish logo placeholder in the content area
    draw.text(
        (width // 2 - 40, height // 2 - 10),
        "Google",
        fill=(70, 130, 180),
        font=font,
    )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


SCREENSHOT_B64 = _make_windowed_browser_screenshot()


def _build_failed_fullscreen_history() -> list[dict]:
    """Build a chat history where the browser is clearly NOT fullscreen.

    The user asked for fullscreen, the screenshot shows a windowed browser
    (visible desktop, title bar, window controls), and the latest notification
    claims ``Computer action executed``.
    """
    return [
        {"role": "user", "content": "Could you please make the browser full screen?"},
        {"role": "assistant", "content": "One moment."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "=== YOUR SCREEN (this is what YOUR machine currently shows) ===",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{SCREENSHOT_B64}",
                    },
                },
            ],
        },
        {
            "role": "system",
            "content": (
                '[notification] Computer action executed: {"thoughts": '
                '"I need to press F11 to make the browser full screen. '
                'This is a simple keyboard action.", "outcome": "executed"}'
            ),
        },
    ]


async def _judge_response(content: str) -> bool:
    """Use an LLM judge to decide if *content* falsely claims fullscreen success.

    Returns True if the response is safe (no false completion claim).
    Returns False if it incorrectly tells the user the browser is fullscreen.
    """
    client = new_llm_client(MODEL, origin="test_judge")
    prompt = (
        "You are a strict test judge. A voice assistant was asked to make a browser "
        "full screen, but the action FAILED — the browser is still in a small window. "
        "The assistant then said the following to the caller:\n\n"
        f'"""{content}"""\n\n'
        "Does this response incorrectly tell the caller that the browser IS full "
        "screen, or that the action was successfully completed? Answer with ONLY "
        '"YES" (it falsely claims success) or "NO" (it does not falsely claim success).'
    )
    verdict = await client.generate(messages=[{"role": "user", "content": prompt}])
    return verdict.strip().upper().startswith("NO")


@pytest.fixture
def evaluator():
    return NotificationReplyEvaluator(model=MODEL)


@pytest.mark.asyncio
async def test_does_not_falsely_confirm_failed_fullscreen(evaluator):
    """When the screenshot shows the browser is NOT fullscreen, the evaluator
    must not produce a confident completion claim like 'All set'.
    """
    chat_history = _build_failed_fullscreen_history()
    decision, _ = await evaluator.evaluate(chat_history=chat_history)

    if decision.speak and decision.content:
        is_safe = await _judge_response(decision.content)
        assert is_safe, (
            f"Evaluator falsely confirmed fullscreen despite screenshot showing "
            f"a windowed browser. Response: {decision.content!r}"
        )


@pytest.mark.asyncio
async def test_confirms_genuinely_completed_action(evaluator):
    """When no screenshot contradicts the notification, the evaluator should
    still be willing to speak a completion confirmation. This ensures the
    prompt change doesn't over-suppress all ``Computer action executed:``
    responses.
    """
    messages: list[dict] = [
        {"role": "user", "content": "Could you open Google for me?"},
        {"role": "assistant", "content": "One moment."},
    ]

    # Screenshot of a fullscreen browser on google.com — no window chrome,
    # no visible desktop. Consistent with the completion claim.
    fullscreen_img = Image.new("RGB", (640, 480), color=(255, 255, 255))
    draw = ImageDraw.Draw(fullscreen_img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            28,
        )
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text((240, 200), "Google", fill=(70, 130, 180), font=font)
    draw.rectangle([200, 150, 440, 170], fill=(240, 240, 240))
    draw.text((210, 152), "google.com", fill=(60, 60, 60), font=font)
    buf = io.BytesIO()
    fullscreen_img.save(buf, format="JPEG", quality=70)
    fullscreen_b64 = base64.b64encode(buf.getvalue()).decode()

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "=== YOUR SCREEN (this is what YOUR machine currently shows) ===",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{fullscreen_b64}",
                    },
                },
            ],
        },
    )
    messages.append(
        {
            "role": "system",
            "content": (
                '[notification] Computer action executed: {"thoughts": '
                '"Opened Chromium and navigated to google.com. The page loaded '
                'successfully.", "outcome": "executed"}'
            ),
        },
    )

    decision, _ = await evaluator.evaluate(chat_history=messages)

    assert decision.speak, (
        "Evaluator should confirm a genuinely completed action when the "
        "screenshot is consistent with the notification."
    )
    assert decision.content, "Expected non-empty speech content."
