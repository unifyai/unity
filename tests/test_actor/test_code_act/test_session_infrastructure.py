import sys

import pytest

from unity.actor.code_act_actor import (
    SessionExecutor,
    _validate_execution_params,
)
from unity.function_manager.function_manager import VenvPool
from unity.function_manager.shell_pool import ShellPool


@pytest.mark.parametrize(
    "kwargs,expect_error_substr",
    [
        (
            dict(
                state_mode="stateless",
                session_id=0,
                session_name=None,
                language="bash",
            ),
            "Cannot use state_mode='stateless' with a session",
        ),
        (
            dict(
                state_mode="read_only",
                session_id=None,
                session_name=None,
                language="python",
            ),
            "Cannot use state_mode='read_only' without specifying a session",
        ),
        (
            dict(
                state_mode="stateless",
                session_id=None,
                session_name=None,
                language="ruby",
            ),
            "Unsupported language",
        ),
        (
            dict(
                state_mode="stateful",
                session_id=1,
                session_name="repo_nav",
                language="bash",
                resolve_session_name=lambda n: (
                    ("bash", None, 0) if n == "repo_nav" else None
                ),
            ),
            "refer to different sessions",
        ),
        (
            dict(
                state_mode="read_only",
                session_id=None,
                session_name="does_not_exist",
                language="python",
                resolve_session_name=lambda _n: None,
            ),
            "not found for read_only",
        ),
        (
            dict(
                state_mode="stateful",
                session_id=None,
                session_name=None,
                language="python",
                max_sessions_total=2,
                active_session_count=2,
            ),
            "Session limit exceeded",
        ),
        (
            dict(
                state_mode="stateful",
                session_id=99,
                session_name=None,
                language="python",
                max_sessions_total=2,
                active_session_count=2,
                session_exists=lambda _l, _v, _s: False,
            ),
            "Session limit exceeded",
        ),
    ],
)
def test_validate_execution_params_matrix(kwargs, expect_error_substr: str):
    err = _validate_execution_params(**kwargs)
    assert isinstance(err, dict)
    assert err.get("error_type") == "validation"
    assert expect_error_substr.lower() in str(err.get("error", "")).lower()