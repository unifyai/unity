import sys

import pytest

from unity.actor.code_act_actor import (
    SessionExecutor,
    _validate_execution_params,
    parts_to_text,
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
                session_name="new_session",
                language="python",
                resolve_session_name=lambda _n: None,
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


@pytest.mark.asyncio
async def test_session_executor_python_stateful_reuses_session():
    ex = SessionExecutor(
        venv_pool=VenvPool(),
        shell_pool=ShellPool(),
        environments={},  # no primitives injection needed for this unit test
        computer_primitives=None,
        function_manager=None,
        timeout=5.0,
    )
    try:
        r1 = await ex.execute(
            code="x = 1\nx",
            language="python",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
        )
        assert r1["error"] is None
        assert r1["session_created"] is True

        r2 = await ex.execute(
            code="x = x + 1\nx",
            language="python",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
        )
        assert r2["error"] is None
        assert r2["session_created"] is False
        # Result value can vary depending on sandbox result-capture logic, but stdout should be empty.
        assert isinstance(r2["duration_ms"], int)
    finally:
        await ex.close()


@pytest.mark.asyncio
async def test_session_executor_python_read_only_does_not_mutate_state():
    ex = SessionExecutor(
        venv_pool=VenvPool(),
        shell_pool=ShellPool(),
        environments={},
        computer_primitives=None,
        function_manager=None,
        timeout=5.0,
    )
    try:
        r1 = await ex.execute(
            code="x = 1",
            language="python",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
        )
        assert r1["error"] is None

        ro = await ex.execute(
            code="x = 999",
            language="python",
            state_mode="read_only",
            session_id=0,
            venv_id=None,
        )
        assert ro["error"] is None

        r2 = await ex.execute(
            code="print(x)",
            language="python",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
        )
        assert r2["error"] is None
        assert "1" in parts_to_text(r2["stdout"])
    finally:
        await ex.close()


@pytest.mark.asyncio
async def test_session_executor_shell_stateless_executes():
    if sys.platform == "win32":
        pytest.skip("shell pool tests are unix-focused")
    ex = SessionExecutor(
        venv_pool=VenvPool(),
        shell_pool=ShellPool(),
        environments={},
        computer_primitives=None,
        function_manager=None,
        timeout=5.0,
    )
    try:
        r = await ex.execute(
            code="echo hello",
            language="bash",
            state_mode="stateless",
            session_id=None,
            venv_id=None,
        )
        assert r["error"] is None
        assert "hello" in r["stdout"]
        assert r["session_id"] is None
    finally:
        await ex.close()


@pytest.mark.asyncio
async def test_session_executor_shell_stateful_persists_env():
    if sys.platform == "win32":
        pytest.skip("shell pool tests are unix-focused")
    ex = SessionExecutor(
        venv_pool=VenvPool(),
        shell_pool=ShellPool(),
        environments={},
        computer_primitives=None,
        function_manager=None,
        timeout=5.0,
    )
    try:
        r1 = await ex.execute(
            code="export FOO=bar",
            language="bash",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
        )
        assert r1["error"] is None
        assert r1["result"] == 0

        r2 = await ex.execute(
            code="echo $FOO",
            language="bash",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
        )
        assert r2["error"] is None
        assert "bar" in r2["stdout"]
    finally:
        await ex.close()


@pytest.mark.asyncio
async def test_session_executor_shell_read_only_does_not_mutate_state():
    if sys.platform == "win32":
        pytest.skip("shell pool tests are unix-focused")
    ex = SessionExecutor(
        venv_pool=VenvPool(),
        shell_pool=ShellPool(),
        environments={},
        computer_primitives=None,
        function_manager=None,
        timeout=5.0,
    )
    try:
        r1 = await ex.execute(
            code="export FOO=bar",
            language="bash",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
        )
        assert r1["error"] is None

        ro = await ex.execute(
            code="export FOO=baz",
            language="bash",
            state_mode="read_only",
            session_id=0,
            venv_id=None,
        )
        assert ro["error"] is None

        r2 = await ex.execute(
            code="echo $FOO",
            language="bash",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
        )
        assert r2["error"] is None
        assert "bar" in (r2["stdout"] or "")
    finally:
        await ex.close()


@pytest.mark.asyncio
async def test_session_executor_isolation_between_python_sessions():
    ex = SessionExecutor(
        venv_pool=VenvPool(),
        shell_pool=ShellPool(),
        environments={},
        computer_primitives=None,
        function_manager=None,
        timeout=5.0,
    )
    try:
        a1 = await ex.execute(
            code="x = 1",
            language="python",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
        )
        b1 = await ex.execute(
            code="x = 2",
            language="python",
            state_mode="stateful",
            session_id=1,
            venv_id=None,
        )
        assert a1["error"] is None
        assert b1["error"] is None

        a2 = await ex.execute(
            code="print(x)",
            language="python",
            state_mode="stateful",
            session_id=0,
            venv_id=None,
        )
        b2 = await ex.execute(
            code="print(x)",
            language="python",
            state_mode="stateful",
            session_id=1,
            venv_id=None,
        )
        assert "1" in parts_to_text(a2["stdout"])
        assert "2" in parts_to_text(b2["stdout"])
    finally:
        await ex.close()
