from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.test_file_manager.helpers import ask_judge


@pytest.mark.asyncio
async def test_basic_ask_and_organize(
    fm_root,
    file_manager,
    global_file_manager,
    tmp_path,
):
    # Ensure singleton Local FileManager is initialized with fm_root
    local = file_manager
    # Create two simple files under fm_root and parse them (no import needed)
    from pathlib import Path

    root = Path(fm_root)
    (root / "g_a.txt").write_text("alpha content")
    (root / "g_b.txt").write_text("beta content")
    f1 = "g_a.txt"
    f2 = "g_b.txt"
    # Parse the files to add them to Unify logs for retrieval operations
    local.ingest_files([f1, f2])

    gfm = global_file_manager

    # Ask at global level (should wire tools and return a handle)
    instruction = "List all of the files. The files are 'g_a.txt' containing 'alpha content' and 'g_b.txt' containing 'beta content'."
    h = await gfm.ask("List all of the files")
    ans = await h.result()
    assert isinstance(ans, str)

    # Judge the response
    verdict = await ask_judge(instruction, ans)
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed 'ask' incorrect. Verdict: {verdict}"

    # Organize at global level to perform a rename
    instruction = f"Rename file {f1} to file_alpha.txt in the local filesystem."
    before_state = {"files": local.list()}
    h2 = await gfm.organize(instruction)
    ans2 = await h2.result()
    assert isinstance(ans2, str)
    after_state = {"files": local.list()}

    # Ask LLM judge to verify the operation
    verdict = await ask_judge(
        instruction,
        ans2,
        before_state=before_state,
        after_state=after_state,
    )
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed rename incorrect. Verdict: {verdict}"

    # Verify rename was executed (keep as a sanity check)
    assert "file_alpha.txt" in after_state["files"]
    assert f1 not in after_state["files"]

    # Sanity: list filesystems helper returns class names
    fs = gfm.list_filesystems()
    assert isinstance(fs, list) and all(isinstance(x, str) for x in fs)


@pytest.mark.asyncio
async def test_organize_rename_and_move(
    fm_root,
    file_manager,
    global_file_manager,
    tmp_path,
):
    from pathlib import Path

    local = file_manager
    gfm = global_file_manager

    # Create test files under fm_root (no import needed)
    src = Path(fm_root)
    dst = Path(fm_root) / "org_dst"  # destination folder under fm_root
    src.mkdir(exist_ok=True)
    dst.mkdir(exist_ok=True)

    (src / "rename_me.txt").write_text("rename me")
    (src / "move_me.txt").write_text("move me")

    a = "rename_me.txt"
    b = "move_me.txt"

    # Parse the files to add them to Unify logs before organize
    local.ingest_files([a, b])

    # Explicit rename via organize prompt
    instruction1 = f"In /local, rename '{a}' to 'renamed.txt' only."
    before_state1 = {"files": local.list()}
    h1 = await gfm.organize(instruction1)
    ans1 = await h1.result()
    after_state1 = {"files": local.list()}

    verdict1 = await ask_judge(
        instruction1,
        ans1,
        before_state=before_state1,
        after_state=after_state1,
    )
    assert (
        verdict1.lower().strip().startswith("correct")
    ), f"Judge deemed rename incorrect. Verdict: {verdict1}"

    # Verify rename reflected in list (keep as sanity check)
    assert "renamed.txt" in after_state1["files"]
    assert a not in after_state1["files"]

    # Explicit move via organize prompt
    instruction2 = f"In /local, move '{b}' into folder 'org_dst' under the same root."
    before_state2 = {"files": local.list()}
    h2 = await gfm.organize(instruction2)
    ans2 = await h2.result()
    after_state2 = {"files": local.list()}

    verdict2 = await ask_judge(
        instruction2,
        ans2,
        before_state=before_state2,
        after_state=after_state2,
    )
    assert (
        verdict2.lower().strip().startswith("correct")
    ), f"Judge deemed move incorrect. Verdict: {verdict2}"

    # Verify move was executed
    # The filename in Unify is now the full path relative to the root
    assert "org_dst/move_me.txt" in after_state2["files"]
    assert b not in after_state2["files"]


@pytest.mark.asyncio
async def test_list_filesystems_and_policy(global_file_manager):
    gfm = global_file_manager
    # Check helper returns class names
    filesystems = gfm.list_filesystems()
    assert isinstance(filesystems, list) and all(
        isinstance(n, str) for n in filesystems
    )

    # Verify organize tool policy requires ask in first step
    with patch(
        "unity.file_manager.global_file_manager.start_async_tool_loop",
    ) as mock_loop:
        mock_handle = MagicMock()
        mock_handle.result = AsyncMock(return_value="ok")
        mock_loop.return_value = mock_handle

        handle = await gfm.organize("noop")
        assert handle is not None
        args, kwargs = mock_loop.call_args
        policy = kwargs.get("tool_policy")
        assert callable(policy)
        mode, tools = policy(
            0,
            {"ask": {"GlobalFileManager_list_filesystems": lambda: None}},
        )
        assert mode == "required" and "ask" in tools


@pytest.mark.asyncio
async def test_organize_delete(
    fm_root,
    file_manager,
    global_file_manager,
    tmp_path,
):
    local = file_manager
    gfm = global_file_manager

    # Create test file under fm_root (no import needed)
    from pathlib import Path

    p = Path(fm_root) / "delete_through_organize.txt"
    p.write_text("delete this via organize")
    filename = "delete_through_organize.txt"
    assert local.exists(filename)

    # Parse the file to add it to Unify logs so we can query for file_id
    local.ingest_files(filename)

    # Get the file_id
    rows = local.filter_files(filter=f"file_path == '{filename}'")
    assert rows
    file_id = rows[0].get("file_id")

    # Delete via the global file manager using organize
    instruction = f"Delete file with id {file_id} from the local filesystem."
    before_state = {"files": local.list()}
    h = await gfm.organize(instruction)
    ans = await h.result()
    after_state = {"files": local.list()}

    # Ask LLM judge to verify the operation
    verdict = await ask_judge(
        instruction,
        ans,
        before_state=before_state,
        after_state=after_state,
    )
    assert (
        verdict.lower().strip().startswith("correct")
    ), f"Judge deemed delete incorrect. Verdict: {verdict}"

    # Verify deletion
    assert filename not in after_state["files"]


@pytest.mark.asyncio
async def test_clarification_integration(global_file_manager):
    gfm = global_file_manager
    with patch(
        "unity.common.clarification_tools.add_clarification_tool_with_events",
    ) as add_clar:
        up_q = AsyncMock()
        down_q = AsyncMock()
        with patch(
            "unity.file_manager.global_file_manager.start_async_tool_loop",
        ) as mock_loop:
            mock_handle = MagicMock()
            mock_handle.result = AsyncMock(return_value="ok")
            mock_loop.return_value = mock_handle
            await gfm.ask(
                "List fs",
                _clarification_up_q=up_q,
                _clarification_down_q=down_q,
            )
            add_clar.assert_called()


@pytest.mark.asyncio
async def test_ask_exposes_class_named_tools(file_manager, global_file_manager):
    gfm = global_file_manager

    with patch(
        "unity.file_manager.global_file_manager.start_async_tool_loop",
    ) as mock_loop:
        mock_handle = MagicMock()
        mock_handle.result = AsyncMock(return_value="ok")
        mock_loop.return_value = mock_handle

        handle = await gfm.ask("List files")
        assert handle is not None

        args, kwargs = mock_loop.call_args
        tools = args[2]
        # Class-named tools should be present
        # We don't know the exact class name in fixture, but must include *_ask and *_ask_about_file
        assert any(k.endswith("_ask") for k in tools.keys())
        assert any(k.endswith("_ask_about_file") for k in tools.keys())

    # Schema should include source_filesystem
    fs = gfm.list_filesystems()
    assert isinstance(fs, list) and fs


@pytest.mark.asyncio
async def test_organize_exposes_class_named_tools(global_file_manager):
    gfm = global_file_manager

    with patch(
        "unity.file_manager.global_file_manager.start_async_tool_loop",
    ) as mock_loop:
        mock_handle = MagicMock()
        mock_handle.result = AsyncMock(return_value="ok")
        mock_loop.return_value = mock_handle

        handle = await gfm.organize("noop")
        assert handle is not None

        args, kwargs = mock_loop.call_args
        tools = args[2]
        # Should contain the discovery tool and class-named organize
        assert "ask" in tools
        assert any(k.endswith("_organize") for k in tools.keys())
