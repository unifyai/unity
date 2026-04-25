from __future__ import annotations

from unity.session_details import SESSION_DETAILS
from unity.transcript_manager.prompt_builders import build_ask_prompt


def _noop(**_kwargs):
    return {}


def _transcript_tools():
    return {
        "filter_messages": _noop,
        "search_messages": _noop,
        "reduce": _noop,
    }


def test_hive_transcript_prompt_frames_peer_authored_rows_as_team_history():
    SESSION_DETAILS.reset()
    try:
        SESSION_DETAILS.populate(agent_id=101, hive_id=77)

        prompt = build_ask_prompt(
            _transcript_tools(),
            num_messages=2,
            transcript_columns={},
            contact_columns={},
        ).flatten()

        assert "team history" in prompt
        assert "my colleague" in prompt
        assert "another assistant on my team" in prompt
        assert "auth_aid" in prompt
        assert "different from 101" in prompt
        assert "normal first-person framing" in prompt
        assert "This body" not in prompt
        assert "another body" not in prompt
    finally:
        SESSION_DETAILS.reset()


def test_solo_transcript_prompt_does_not_render_team_history_block():
    SESSION_DETAILS.reset()
    try:
        SESSION_DETAILS.populate(agent_id=101, hive_id=None)

        prompt = build_ask_prompt(
            _transcript_tools(),
            num_messages=2,
            transcript_columns={},
            contact_columns={},
        ).flatten()

        assert "team history" not in prompt
        assert "my colleague" not in prompt
    finally:
        SESSION_DETAILS.reset()
