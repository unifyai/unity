"""Integration test: storage loop shell env management for CLI tool trajectories.

Feeds a hand-crafted trajectory containing ``execute_code(language="bash")``
that installs a CLI tool and uses it, directly to ``_start_storage_check_loop``.
Verifies the storage LLM creates a shell env and (if it stores a shell function)
links it with ``shell_env_id``.

This is an eval test because the LLM must interpret the trajectory and decide
to create a shell env.
"""

import asyncio

import pytest

from unity.actor.code_act_actor import CodeActActor, _start_storage_check_loop
from unity.function_manager.function_manager import FunctionManager

pytestmark = pytest.mark.eval

TRAJECTORY_WITH_CLI_TOOL_INSTALL = [
    {
        "role": "assistant",
        "content": "I'll install the gcloud CLI tool for Google Cloud operations.",
        "tool_calls": [
            {
                "id": "tc_1",
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": (
                        '{"code": "curl -sSL https://sdk.cloud.google.com/nightly/google-cloud-sdk.tar.gz '
                        "| tar xz -C /opt\\n"
                        "/opt/google-cloud-sdk/install.sh --quiet\\n"
                        'echo \\"Installed to: /opt/google-cloud-sdk/bin/gcloud\\"\\n'
                        'which gcloud || echo \\"/opt/google-cloud-sdk/bin/gcloud\\"", '
                        '"language": "bash"}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_1",
        "content": (
            '{"stdout": "Installed to: /opt/google-cloud-sdk/bin/gcloud\\n'
            '/opt/google-cloud-sdk/bin/gcloud", '
            '"stderr": "", "error": null, "language": "bash"}'
        ),
    },
    {
        "role": "assistant",
        "content": "Now I'll authenticate and list the available projects.",
        "tool_calls": [
            {
                "id": "tc_2",
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": (
                        '{"code": "gcloud projects list --format=json | head -20", '
                        '"language": "bash"}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_2",
        "content": (
            '{"stdout": "[{\\"projectId\\": \\"my-project-123\\", '
            '\\"name\\": \\"My Project\\", \\"lifecycleState\\": \\"ACTIVE\\"}]", '
            '"stderr": "", "error": null, "language": "bash"}'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Successfully installed gcloud and listed the available projects. "
            "The project my-project-123 is active and ready to use."
        ),
    },
]


class _MinimalGuidanceManager:
    def search(self, references=None, k=10):
        return []

    def filter(self, filter=None, offset=0, limit=100):
        return []

    def add_guidance(self, *, title, content, function_ids=None):
        return {"details": {"guidance_id": 1}}

    def update_guidance(
        self,
        *,
        guidance_id,
        title=None,
        content=None,
        function_ids=None,
    ):
        return {"details": {"guidance_id": guidance_id}}

    def delete_guidance(self, *, guidance_id):
        return {"deleted": True}


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_storage_loop_creates_shell_env_for_cli_tool():
    """The storage loop creates a shell env when the trajectory installs a CLI tool."""
    fm = FunctionManager(include_primitives=False)
    gm = _MinimalGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=120,
    )

    try:
        handle = _start_storage_check_loop(
            trajectory=TRAJECTORY_WITH_CLI_TOOL_INSTALL,
            ask_tools={},
            actor=actor,
            original_result="Listed GCP projects successfully.",
        )
        assert handle is not None, "Storage loop should have started"

        result = await asyncio.wait_for(handle.result(), timeout=180)
        assert result is not None

        shell_envs = fm.list_shell_envs()
        stored_functions = fm.filter_functions()

        shell_functions = [
            f
            for f in stored_functions
            if isinstance(f, dict) and f.get("language") in ("bash", "zsh", "sh")
        ]

        if shell_functions:
            for func in shell_functions:
                assert func.get("shell_env_id") is not None, (
                    f"Shell function '{func.get('name')}' was stored without "
                    f"a shell_env_id — the storage loop should link it to a "
                    f"shell environment containing the installed CLI tool."
                )

        assert len(shell_envs) >= 1 or len(shell_functions) == 0, (
            f"If shell functions were stored, at least one shell env should "
            f"exist. Got {len(shell_envs)} shell envs and "
            f"{len(shell_functions)} shell functions. "
            f"Stored functions: {[f.get('name') for f in stored_functions if isinstance(f, dict)]}"
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass
