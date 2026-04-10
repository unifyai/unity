"""Integration tests: storage loop venv management for third-party imports.

Two tests:

1. **Synthetic trajectory** — feeds a hand-crafted trajectory containing
   ``install_python_packages`` + ``execute_code`` with third-party imports
   directly to ``_start_storage_check_loop``.  Verifies the storage LLM
   creates a venv and stores the function with ``venv_id`` set.

2. **Full act() end-to-end** — calls ``actor.act()`` with ``can_store=True``
   on a task that requires ``install_python_packages`` and writes a reusable
   function.  The doing loop installs the package, executes code, and the
   ``_StorageCheckHandle`` fires the storage review automatically.  Verifies
   the stored function has ``venv_id`` set and at least one venv was created.

Both are eval tests because the LLM must correctly interpret the rejection
error and decide to create a venv.
"""

import asyncio

import pytest

from unity.actor.code_act_actor import CodeActActor, _start_storage_check_loop
from unity.function_manager.function_manager import FunctionManager

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]

TRAJECTORY_WITH_THIRD_PARTY_PACKAGES = [
    {
        "role": "assistant",
        "content": "I'll install the required packages for Google Cloud Storage operations.",
        "tool_calls": [
            {
                "id": "tc_1",
                "type": "function",
                "function": {
                    "name": "install_python_packages",
                    "arguments": '{"packages": ["google-cloud-storage>=2.10.0"]}',
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_1",
        "content": (
            '{"success": true, "stdout": "Installed google-cloud-storage and dependencies", '
            '"stderr": "", "packages": ["google-cloud-storage>=2.10.0"]}'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Now I'll write and test a comprehensive GCS file management utility "
            "that handles uploads with retry logic, metadata management, and "
            "signed URL generation."
        ),
        "tool_calls": [
            {
                "id": "tc_2",
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": (
                        '{"code": "'
                        "import json\\n"
                        "import datetime\\n"
                        "\\n"
                        "async def manage_gcs_files(\\n"
                        "    bucket_name: str,\\n"
                        "    operation: str,\\n"
                        "    blob_path: str,\\n"
                        "    *,\\n"
                        "    local_path: str = None,\\n"
                        "    content_type: str = None,\\n"
                        "    metadata: dict = None,\\n"
                        "    signed_url_expiry_hours: int = 1,\\n"
                        "    max_retries: int = 3,\\n"
                        ") -> dict:\\n"
                        '    \\"\\"\\"Manage files in Google Cloud Storage with retry logic and metadata.\\n'
                        "    \\n"
                        "    Supports upload, download, delete, get_metadata, and generate_signed_url\\n"
                        "    operations. Handles transient failures with exponential backoff.\\n"
                        "    \\n"
                        "    Args:\\n"
                        "        bucket_name: GCS bucket name.\\n"
                        "        operation: One of upload, download, delete, get_metadata, signed_url.\\n"
                        "        blob_path: Path of the blob within the bucket.\\n"
                        "        local_path: Local file path for upload/download operations.\\n"
                        "        content_type: MIME type for uploads (auto-detected if not provided).\\n"
                        "        metadata: Custom metadata dict to attach on upload.\\n"
                        "        signed_url_expiry_hours: Expiry for signed URLs (default 1 hour).\\n"
                        "        max_retries: Max retry attempts for transient failures.\\n"
                        "    \\n"
                        "    Returns:\\n"
                        "        Dict with operation status, blob details, and any generated URLs.\\n"
                        '    \\"\\"\\"\\n'
                        "    import time\\n"
                        "    from google.cloud import storage\\n"
                        "    from google.api_core import exceptions as gcs_exceptions\\n"
                        "    \\n"
                        "    client = storage.Client()\\n"
                        "    bucket = client.bucket(bucket_name)\\n"
                        "    blob = bucket.blob(blob_path)\\n"
                        "    \\n"
                        "    for attempt in range(max_retries):\\n"
                        "        try:\\n"
                        '            if operation == \\"upload\\":\\n'
                        "                if local_path is None:\\n"
                        '                    raise ValueError(\\"local_path required for upload\\")\\n'
                        "                blob.upload_from_filename(local_path, content_type=content_type)\\n"
                        "                if metadata:\\n"
                        "                    blob.metadata = metadata\\n"
                        "                    blob.patch()\\n"
                        '                return {\\"status\\": \\"uploaded\\", \\"blob\\": blob_path, \\"size\\": blob.size}\\n'
                        '            elif operation == \\"download\\":\\n'
                        "                if local_path is None:\\n"
                        '                    raise ValueError(\\"local_path required for download\\")\\n'
                        "                blob.download_to_filename(local_path)\\n"
                        '                return {\\"status\\": \\"downloaded\\", \\"blob\\": blob_path, \\"local\\": local_path}\\n'
                        '            elif operation == \\"delete\\":\\n'
                        "                blob.delete()\\n"
                        '                return {\\"status\\": \\"deleted\\", \\"blob\\": blob_path}\\n'
                        '            elif operation == \\"get_metadata\\":\\n'
                        "                blob.reload()\\n"
                        "                return {\\n"
                        '                    \\"status\\": \\"ok\\", \\"blob\\": blob_path,\\n'
                        '                    \\"size\\": blob.size, \\"content_type\\": blob.content_type,\\n'
                        '                    \\"metadata\\": blob.metadata or {},\\n'
                        '                    \\"updated\\": str(blob.updated),\\n'
                        "                }\\n"
                        '            elif operation == \\"signed_url\\":\\n'
                        "                url = blob.generate_signed_url(\\n"
                        "                    expiration=datetime.timedelta(hours=signed_url_expiry_hours),\\n"
                        '                    method=\\"GET\\",\\n'
                        "                )\\n"
                        '                return {\\"status\\": \\"ok\\", \\"url\\": url, \\"expires_hours\\": signed_url_expiry_hours}\\n'
                        "            else:\\n"
                        '                raise ValueError(f\\"Unknown operation: {operation}\\")\\n'
                        "        except (gcs_exceptions.ServiceUnavailable, gcs_exceptions.TooManyRequests) as e:\\n"
                        "            if attempt == max_retries - 1:\\n"
                        "                raise\\n"
                        "            time.sleep(2 ** attempt)\\n"
                        "\\n"
                        'result = await manage_gcs_files(\\"test-bucket\\", \\"get_metadata\\", \\"data/report.csv\\")\\n'
                        "print(json.dumps(result, indent=2))"
                        '"}'
                    ),
                },
            },
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "tc_2",
        "content": (
            '{"stdout": "{\\n  \\"status\\": \\"ok\\",\\n  \\"blob\\": \\"data/report.csv\\",\\n  '
            '\\"size\\": 45231,\\n  \\"content_type\\": \\"text/csv\\",\\n  \\"metadata\\": '
            '{\\"owner\\": \\"analytics\\"},\\n  \\"updated\\": \\"2026-03-08 12:30:00\\"\\n}\\n", '
            '"stderr": "", "success": true}'
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The GCS file management utility works correctly. It supports upload, "
            "download, delete, metadata retrieval, and signed URL generation with "
            "automatic retry logic for transient failures."
        ),
    },
]


class _MinimalGuidanceManager:
    def search(self, references=None, k=10):
        """Search for guidance entries."""
        return []

    def filter(self, filter=None, offset=0, limit=100):
        """Filter guidance entries."""
        return []

    def add_guidance(self, *, title, content, function_ids=None):
        """Add a guidance entry."""
        return {"details": {"guidance_id": 1}}

    def update_guidance(
        self,
        *,
        guidance_id,
        title=None,
        content=None,
        function_ids=None,
    ):
        """Update guidance."""
        return {"details": {"guidance_id": guidance_id}}

    def delete_guidance(self, *, guidance_id):
        """Delete guidance."""
        return {"deleted": True}


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_storage_loop_creates_venv_for_third_party_function():
    """The storage loop creates a venv and links it to a function that uses requests."""
    fm = FunctionManager(include_primitives=False)
    gm = _MinimalGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=120,
    )

    try:
        handle = _start_storage_check_loop(
            trajectory=TRAJECTORY_WITH_THIRD_PARTY_PACKAGES,
            ask_tools={},
            actor=actor,
            original_result="Fetched JSON from httpbin.org successfully.",
        )
        assert handle is not None, "Storage loop should have started"

        result = await asyncio.wait_for(handle.result(), timeout=180)
        assert result is not None

        stored = fm.filter_functions()
        stored_with_tp = [
            f for f in stored if isinstance(f, dict) and f.get("third_party_imports")
        ]
        assert stored_with_tp, (
            f"Expected at least one function with third_party_imports to be "
            f"stored. All stored functions: "
            f"{[f.get('name') for f in stored if isinstance(f, dict)]}"
        )

        for func in stored_with_tp:
            assert func.get("venv_id") is not None, (
                f"Function '{func.get('name')}' has third-party imports "
                f"{func.get('third_party_imports')} but venv_id is None"
            )

        venvs = fm.list_venvs()
        assert (
            len(venvs) >= 1
        ), f"Expected at least one venv to be created, got {len(venvs)}"
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Full act() end-to-end: doing loop installs package, storage loop creates venv
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_act_with_package_install_stores_function_with_venv():
    """Full end-to-end: act() installs a package, writes a function, and the
    storage check loop creates a venv and links it.

    The task is deliberately designed so the doing loop must:
    1. ``install_python_packages(["humanize"])``
    2. Write and execute a reusable function that ``import humanize``
    3. Complete successfully

    Then the implicit ``_StorageCheckHandle`` fires and the storage librarian
    must recognise the third-party dependency, create a venv, and store the
    function with ``venv_id`` set.

    ``humanize`` is a lightweight, pure-Python package that installs fast and
    has a distinctive API (``humanize.naturalsize``, ``humanize.naturaltime``)
    making the function clearly non-trivial and worth storing.
    """
    fm = FunctionManager(include_primitives=False)
    gm = _MinimalGuidanceManager()

    actor = CodeActActor(
        function_manager=fm,
        guidance_manager=gm,
        timeout=180,
    )
    try:
        handle = await actor.act(
            "Install the `humanize` Python package, then write and execute a "
            "reusable function called `format_data_size` that:\n\n"
            "1. Takes `size_bytes: int` and an optional `gnu: bool = False`\n"
            "2. Uses `humanize.naturalsize` to format the byte count into a "
            "human-readable string (e.g. '1.5 GB', '45.2 kB')\n"
            "3. When `gnu=True`, uses GNU-style units (GiB, MiB)\n"
            "4. Handles edge cases: negative sizes (format absolute value "
            "with a '-' prefix), zero (returns '0 Bytes')\n"
            "5. Returns the formatted string\n\n"
            "Test it with these inputs:\n"
            "- 0 -> '0 Bytes'\n"
            "- 1024 -> should contain 'kB' or 'KB'\n"
            "- 1073741824 -> should contain 'GB' or 'GiB'\n"
            "- -500 -> should start with '-'\n\n"
            "Make sure the function is correct before finishing.",
            can_store=True,
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=300)
        assert result is not None

        deadline = asyncio.get_event_loop().time() + 180
        while not handle.done():
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("Storage loop did not complete in time")
            await asyncio.sleep(0.5)

        stored = fm.filter_functions()
        stored_with_tp = [
            f for f in stored if isinstance(f, dict) and f.get("third_party_imports")
        ]
        assert stored_with_tp, (
            f"Expected at least one function with third_party_imports to be "
            f"stored after a full act() that installed humanize. "
            f"All stored: {[f.get('name') for f in stored if isinstance(f, dict)]}"
        )

        for func in stored_with_tp:
            assert func.get("venv_id") is not None, (
                f"Function '{func.get('name')}' has third-party imports "
                f"{func.get('third_party_imports')} but venv_id is None — "
                f"the storage loop failed to create and link a venv."
            )

        venvs = fm.list_venvs()
        assert len(venvs) >= 1, (
            f"Expected at least one venv to be created by the storage loop, "
            f"got {len(venvs)}"
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass
