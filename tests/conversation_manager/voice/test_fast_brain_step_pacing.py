"""
Regression tests for fast brain step-by-step pacing during live walkthroughs.

Production issue: when the user is actively working through a multi-step guide
one click at a time (on a Unify Meet screen-share), the fast brain chains
multiple UI actions into a single response — e.g. "click X, then search Y,
then press ENABLE". The user repeatedly said "going too fast again".

Key behavioural rule:
- An initial succinct overview of the steps ahead is *fine*.
- But once the user enters "execution mode" — completing steps and asking for
  the next one, or asking for a step to be repeated — every response must
  contain exactly ONE user action. Think of it like a human saying "now click
  X" and then *waiting* for "okay, done" before continuing.

Ref: production logs from unity-2026-03-09-02-18-32-uda01-staging
"""

from __future__ import annotations

import re

import pytest

from unity.conversation_manager.prompt_builders import build_voice_agent_prompt

from tests.conversation_manager.voice.test_fast_brain_deferral import (
    MODEL_TTS,
    get_fast_brain_response,
)

pytestmark = pytest.mark.eval


MULTI_STEP_NOTIFICATION = (
    "[notification] Action update: \n"
    "Based on the web research and the user's current screen, here are the "
    "exact steps from where they are now:\n\n"
    "## Step-by-step from the Google Cloud Console\n\n"
    "### Step 1: Enable the Google Drive API\n"
    '1. Click "APIs & Services" in the Quick Access section\n'
    '2. Click "+ ENABLE APIS AND SERVICES" (blue button at the top)\n'
    '3. In the API Library search bar, type "Google Drive API"\n'
    '4. Click on "Google Drive API" in the results\n'
    '5. Click the blue "ENABLE" button\n\n'
    "### Step 2: Configure the OAuth Consent Screen\n"
    '1. In the left sidebar, click "OAuth consent screen"\n'
    '2. Select "External" as the user type, then click "CREATE"\n'
    "3. Fill in the required fields:\n"
    '   - App name: anything (e.g. "My Drive Assistant")\n'
    "   - User support email: select from dropdown\n"
    "   - Developer contact email: enter email address\n"
    '4. Click "SAVE AND CONTINUE"\n'
    '5. On the Scopes screen — click "ADD OR REMOVE SCOPES", find Google '
    "Drive API, check the box, click UPDATE, then SAVE AND CONTINUE\n"
    '6. On the Test users screen — click "+ ADD USERS", enter Gmail address, '
    'click ADD, then "SAVE AND CONTINUE"\n'
    '7. Click "BACK TO DASHBOARD"\n\n'
    "### Step 3: Create OAuth 2.0 Client Credentials\n"
    '1. In the left sidebar, click "Credentials"\n'
    '2. Click "+ CREATE CREDENTIALS" then choose "OAuth client ID"\n'
    '3. For Application type, select "Desktop app"\n'
    '4. Give it a name (e.g. "Drive Desktop Client")\n'
    '5. Click "CREATE"\n\n'
    "### Step 4: What to copy\n"
    "A popup will appear showing:\n"
    "- Client ID — a long string ending in .apps.googleusercontent.com\n"
    "- Client Secret — a shorter string starting with GOCSPX-\n\n"
    "Right now, the immediate instruction for the user is: "
    'Click "APIs & Services" on the dashboard.'
)


def _build_meet_prompt() -> str:
    return build_voice_agent_prompt(
        bio="A helpful and efficient assistant.",
        assistant_name="Alex",
        boss_first_name="Dan",
        boss_surname="Sherwood",
        boss_phone_number="+447700900000",
        boss_email_address="dan@example.com",
        boss_bio="Non-technical user. Prefers clear, one-step-at-a-time guidance.",
        is_boss_user=True,
        contact_rolling_summary=None,
    ).flatten()


# UI-action verbs that indicate a distinct step the user must perform.
_ACTION_VERBS = re.compile(
    r"\b(?:click|tap|press|open|select|choose|type|enter|search|enable|"
    r"disable|check|uncheck|toggle|drag|scroll|navigate|go to|switch to|"
    r"pick|hit|find)\b",
    re.IGNORECASE,
)


_NOUN_PHRASE_VERBS = re.compile(
    r"\bsearch (results?|bar|box|field|page)\b"
    r"|\bcheck ?box\b"
    r"|\bselect(ed|ion) \b"
    r"|\bopen(ed|ing)? (tab|window|page|panel|source)\b"
    r"|\bnavigate to\b",
    re.IGNORECASE,
)


def _count_ui_actions(response: str) -> int:
    """Count distinct UI actions described in the response.

    Catches compound instructions like 'click X, then type Y, then press Z'
    that bypass numbered-list detection.

    Strips quoted strings and known noun-phrase uses of action verbs so
    that button labels ('"ENABLE APIS AND SERVICES"') and incidental
    phrases ("in the search results") don't inflate the count.
    """
    stripped = re.sub(r'"[^"]*"', '""', response)
    stripped = re.sub(r"\u201c[^\u201d]*\u201d", '""', stripped)
    stripped = _NOUN_PHRASE_VERBS.sub("_NP_", stripped)
    return len(_ACTION_VERBS.findall(stripped))


class TestStepPacingDuringExecution:
    """Once the user is actively executing steps (saying 'done, what next?'),
    every response must contain exactly one user action."""

    @pytest.mark.asyncio
    async def test_one_action_after_completing_first_step(self):
        """User completed step 1 and says 'done, what's next?' — the fast brain
        must give exactly one next action, not chain the remaining plan.

        Regression: the fast brain responded with 'click ENABLE APIS AND
        SERVICES, search for Google Drive API, open it, and click ENABLE'
        (6 actions in one turn).
        """
        prompt = _build_meet_prompt()
        conversation = [
            {
                "role": "user",
                "content": (
                    "I want to set up Google Drive access. I'm on the "
                    "Google Cloud Console welcome page."
                ),
            },
            {"role": "assistant", "content": "Let me look into that for you."},
            {"role": "system", "content": MULTI_STEP_NOTIFICATION},
            {
                "role": "assistant",
                "content": (
                    'Click "APIs & Services" in the Quick Access section on '
                    "the dashboard."
                ),
            },
            {"role": "user", "content": "Okay, done. What's next?"},
        ]

        response = await get_fast_brain_response(prompt, conversation, model=MODEL_TTS)
        actions = _count_ui_actions(response)

        assert actions <= 1, (
            f"Fast brain described {actions} UI actions — must be exactly 1.\n"
            f"The user just completed a step and asked 'what's next?'. A human "
            f"guide would say the single next action and wait for confirmation.\n"
            f"Response:\n{response}"
        )

    @pytest.mark.asyncio
    async def test_one_action_when_user_asks_to_repeat(self):
        """When the user asks for the current step to be repeated, the response
        must contain only that single action — not the current step plus what
        comes after.

        Regression: when the user said 'sorry, what was that again?', the fast
        brain repeated the step *and* appended the next 2-3 steps.
        """
        prompt = _build_meet_prompt()
        conversation = [
            {
                "role": "user",
                "content": "I'm setting up Google OAuth. I'm on the console.",
            },
            {"role": "assistant", "content": "Let me look that up."},
            {"role": "system", "content": MULTI_STEP_NOTIFICATION},
            {
                "role": "assistant",
                "content": (
                    'Click "APIs & Services" in Quick Access, then click '
                    '"+ ENABLE APIS AND SERVICES" at the top.'
                ),
            },
            {
                "role": "user",
                "content": (
                    "Hang on, you're going too fast. What was the first thing "
                    "I need to click?"
                ),
            },
        ]

        response = await get_fast_brain_response(prompt, conversation, model=MODEL_TTS)
        actions = _count_ui_actions(response)

        assert actions <= 1, (
            f"Fast brain described {actions} UI actions — must be exactly 1.\n"
            f"The user explicitly said 'going too fast' and asked for the step "
            f"to be repeated. Give only that single action.\n"
            f"Response:\n{response}"
        )

    @pytest.mark.asyncio
    async def test_one_action_on_confused_user_followup(self):
        """When the user signals confusion ('I don't see that', 'where is it?'),
        the response should re-explain the current action — not pile on more
        steps.
        """
        prompt = _build_meet_prompt()
        conversation = [
            {
                "role": "user",
                "content": "I'm on the Google Cloud Console, setting up OAuth.",
            },
            {"role": "assistant", "content": "Let me check on that."},
            {"role": "system", "content": MULTI_STEP_NOTIFICATION},
            {
                "role": "assistant",
                "content": 'Click "OAuth consent screen" in the left sidebar.',
            },
            {
                "role": "user",
                "content": "I don't see an OAuth consent screen in the sidebar.",
            },
        ]

        response = await get_fast_brain_response(prompt, conversation, model=MODEL_TTS)
        actions = _count_ui_actions(response)

        assert actions <= 2, (
            f"Fast brain described {actions} UI actions — should be at most 2.\n"
            f"The user is confused about the current step. Re-explain how to "
            f"find it (possibly with one alternative path), but don't jump "
            f"ahead to future steps.\n"
            f"Response:\n{response}"
        )
