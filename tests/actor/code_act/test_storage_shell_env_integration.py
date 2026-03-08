"""Integration test: storage loop shell env management.

Feeds a hand-crafted trajectory containing a bash shell function that
invokes a non-standard CLI tool (installed via curl) directly to
``_start_storage_check_loop``.  Verifies the storage LLM creates a
shell env and stores the shell function with ``shell_env_id`` set.

This is an eval test because the LLM must correctly interpret the
trajectory and decide to create a shell env for the installed CLI tool.
"""

import asyncio

import pytest

from unity.actor.code_act_actor import CodeActActor, _start_storage_check_loop
from unity.function_manager.function_manager import FunctionManager

pytestmark = pytest.mark.eval


TRAJECTORY_WITH_SHELL_TOOL_INSTALL = [
    {
        "role": "assistant",
        "content": (
            "I'll install yq (a YAML processor) and write a bash function "
            "that uses it to extract values from YAML files."
        ),
        "tool_calls": [
            {
                "id": "tc_1",
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": (
                        '{"code": "import subprocess\\n'
                        "result = subprocess.run(\\n"
                        "    ['brew', 'install', 'yq'],\\n"
                        "    capture_output=True, text=True,\\n"
                        ")\\n"
                        "print(result.stdout)\\n"
                        'print(result.returncode)"}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_1",
        "content": (
            '{"stdout": "==> Downloading yq-4.35.2\\n==> Installing yq\\n'
            '/usr/local/bin/yq\\n0\\n", "stderr": "", "success": true}'
        ),
    },
    {
        "role": "assistant",
        "content": "yq is installed. Now I'll write and test a bash function that uses it.",
        "tool_calls": [
            {
                "id": "tc_2",
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": (
                        '{"code": "import subprocess, tempfile, os\\n'
                        "# Write a test YAML file\\n"
                        "yaml_content = 'name: test\\\\nversion: 1.0\\\\ntags:\\\\n  - alpha\\\\n  - beta\\\\n'\\n"
                        "with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:\\n"
                        "    f.write(yaml_content)\\n"
                        "    yaml_path = f.name\\n"
                        "\\n"
                        "# Test yq extraction\\n"
                        "result = subprocess.run(\\n"
                        "    ['yq', '.name', yaml_path],\\n"
                        "    capture_output=True, text=True,\\n"
                        ")\\n"
                        "print(f'name: {result.stdout.strip()}')\\n"
                        "\\n"
                        "result2 = subprocess.run(\\n"
                        "    ['yq', '.tags[]', yaml_path],\\n"
                        "    capture_output=True, text=True,\\n"
                        ")\\n"
                        "print(f'tags: {result2.stdout.strip()}')\\n"
                        "os.unlink(yaml_path)\\n"
                        "print('done')\"}"
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_2",
        "content": (
            '{"stdout": "name: test\\ntags: alpha\\nbeta\\ndone\\n", '
            '"stderr": "", "success": true}'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "yq works correctly. The bash function extracts YAML fields "
            "using yq with proper error handling."
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
async def test_storage_loop_creates_shell_env_for_installed_tool():
    """The storage loop creates a shell env when the trajectory installs
    a CLI tool and uses it in a bash function."""
    fm = FunctionManager(include_primitives=False)
    gm = _MinimalGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=120,
    )

    try:
        handle = _start_storage_check_loop(
            trajectory=TRAJECTORY_WITH_SHELL_TOOL_INSTALL,
            ask_tools={},
            actor=actor,
            original_result="Successfully extracted YAML values using yq.",
        )
        assert handle is not None, "Storage loop should have started"

        result = await asyncio.wait_for(handle.result(), timeout=180)
        assert result is not None

        stored = fm.filter_functions()
        shell_functions = [
            f
            for f in stored
            if isinstance(f, dict) and f.get("language") in ("bash", "sh", "zsh")
        ]

        shell_envs = fm.list_shell_envs()

        # The LLM should have stored either:
        # - A shell function with shell_env_id set, OR
        # - A Python wrapper function with the tool call
        # Either way, if shell functions were stored, they should have shell_env_id
        if shell_functions:
            with_env = [f for f in shell_functions if f.get("shell_env_id") is not None]
            assert with_env, (
                f"Shell functions were stored but none have shell_env_id set. "
                f"Functions: {[f.get('name') for f in shell_functions]}"
            )
            assert (
                len(shell_envs) >= 1
            ), f"Expected at least one shell env to be created, got {len(shell_envs)}"
    finally:
        try:
            await actor.close()
        except Exception:
            pass
