"""Integration tests: storage loop shell function and shell env management.

Two tests:

1. **Synthetic trajectory (symbolic)** — feeds a hand-crafted trajectory
   containing bash script execution with an installed CLI tool directly to
   ``_start_storage_check_loop``.  Verifies the storage LLM stores a
   function (Python or shell) that captures the reusable pattern.

2. **Synthetic trajectory with explicit shell script** — trajectory where
   the agent wrote and executed a bash script using an installed tool.
   Verifies that the storage LLM recognises this as a shell function
   candidate and stores it with the correct language.

Both are eval tests because the LLM must correctly interpret the trajectory
and decide what to store.
"""

import asyncio

import pytest

from unity.actor.code_act_actor import CodeActActor, _start_storage_check_loop
from unity.function_manager.function_manager import FunctionManager

pytestmark = pytest.mark.eval


# Trajectory where the agent installs jq and writes a reusable shell script
TRAJECTORY_WITH_SHELL_SCRIPT = [
    {
        "role": "assistant",
        "content": (
            "I'll install jq for JSON processing and create a reusable "
            "bash script that extracts nested fields from JSON files."
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
                        "    ['brew', 'install', 'jq'],\\n"
                        "    capture_output=True, text=True,\\n"
                        ")\\n"
                        "print(result.stdout)\\n"
                        "print('exit:', result.returncode)\"}"
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_1",
        "content": (
            '{"stdout": "==> Downloading jq-1.7.1\\n'
            "==> Installing jq\\n"
            "/usr/local/bin/jq\\n"
            'exit: 0\\n", "stderr": "", "success": true}'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "jq is installed at /usr/local/bin/jq. Now I'll write a bash "
            "script that uses jq to extract nested JSON fields and test it."
        ),
        "tool_calls": [
            {
                "id": "tc_2",
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": (
                        '{"code": "import subprocess, tempfile, json, os\\n'
                        "\\n"
                        "# Write test JSON\\n"
                        "test_data = {'users': [{'name': 'Alice', 'email': 'alice@example.com'}, {'name': 'Bob', 'email': 'bob@example.com'}]}\\n"
                        "with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:\\n"
                        "    json.dump(test_data, f)\\n"
                        "    json_path = f.name\\n"
                        "\\n"
                        "# Write and test a bash script that uses jq\\n"
                        "script = '''#!/bin/bash\\n"
                        "# @name: extract_json_emails\\n"
                        "# @args: (json_file)\\n"
                        "# @description: Extract all email addresses from a JSON file using jq\\n"
                        "jq -r '.users[].email' \\\"$1\\\"\\n"
                        "'''\\n"
                        "\\n"
                        "with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:\\n"
                        "    f.write(script)\\n"
                        "    script_path = f.name\\n"
                        "os.chmod(script_path, 0o755)\\n"
                        "\\n"
                        "result = subprocess.run(\\n"
                        "    ['bash', script_path, json_path],\\n"
                        "    capture_output=True, text=True,\\n"
                        ")\\n"
                        "print(f'emails: {result.stdout.strip()}')\\n"
                        "print(f'exit: {result.returncode}')\\n"
                        "os.unlink(json_path)\\n"
                        'os.unlink(script_path)"}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_2",
        "content": (
            '{"stdout": "emails: alice@example.com\\nbob@example.com\\n'
            'exit: 0\\n", "stderr": "", "success": true}'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The bash script successfully extracts email addresses from "
            "JSON files using jq. The script is reusable for any JSON file "
            "with a .users[].email structure."
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
async def test_storage_loop_stores_function_from_shell_trajectory():
    """The storage loop stores at least one function from a trajectory
    that installed a CLI tool and used it in a bash script.

    The LLM may store either a Python wrapper or a shell function —
    both are valid. The key assertion is that *something* reusable
    was extracted from the trajectory.
    """
    fm = FunctionManager(include_primitives=False)
    gm = _MinimalGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=120,
    )

    try:
        handle = _start_storage_check_loop(
            trajectory=TRAJECTORY_WITH_SHELL_SCRIPT,
            ask_tools={},
            actor=actor,
            original_result=(
                "Successfully extracted email addresses from JSON using jq."
            ),
        )
        assert handle is not None, "Storage loop should have started"

        result = await asyncio.wait_for(handle.result(), timeout=180)
        assert result is not None

        stored = fm.filter_functions()
        stored_funcs = [f for f in stored if isinstance(f, dict)]

        assert stored_funcs, (
            "Expected at least one function to be stored from a trajectory "
            "that installed jq and wrote a reusable bash script."
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass
