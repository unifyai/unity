from __future__ import annotations

import pytest

from tests.test_file_manager.helpers import ask_judge


@pytest.mark.asyncio
async def test_global_fm_basic_ask_and_organize(
    fm_root,
    file_manager,
    global_file_manager,
    tmp_path,
):
    # Ensure singleton Local FileManager is initialized with fm_root
    local = file_manager
    # Import two simple files

    # Create test files OUTSIDE fm_root to avoid duplication on import
    p1 = tmp_path / "g_a.txt"
    p1.write_text("alpha content")
    p2 = tmp_path / "g_b.txt"
    p2.write_text("beta content")
    f1 = local.import_file(p1)
    f2 = local.import_file(p2)

    # Parse the files to add them to Unify logs for retrieval operations
    local.parse([f1, f2])

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

    # Aggregated retrieval helpers
    cols = gfm._list_columns()
    assert "source_filesystem" in cols

    res = gfm._filter_files()
    assert isinstance(res, list)
    assert any(r.get("filename", "").startswith("/local/") for r in res)

    sr = gfm._search_files(references={"full_text": "alpha"}, k=3)
    assert isinstance(sr, list)
    if sr:
        assert sr[0]["filename"].startswith("/local/")


@pytest.mark.asyncio
async def test_global_fm_organize_rename_and_move(
    fm_root,
    file_manager,
    global_file_manager,
    tmp_path,
):
    from pathlib import Path

    local = file_manager
    gfm = global_file_manager

    # Create test files OUTSIDE fm_root to avoid duplication on import
    src = tmp_path / "org_src"
    dst = Path(fm_root) / "org_dst"  # dst stays in fm_root as it's the target
    src.mkdir(exist_ok=True)
    dst.mkdir(exist_ok=True)

    (src / "rename_me.txt").write_text("rename me")
    (src / "move_me.txt").write_text("move me")

    a = local.import_file(src / "rename_me.txt")
    b = local.import_file(src / "move_me.txt")

    # Parse the files to add them to Unify logs before organize
    local.parse([a, b])

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
async def test_global_fm_delete_file(
    fm_root,
    file_manager,
    global_file_manager,
    tmp_path,
):
    pass

    local = file_manager
    gfm = global_file_manager

    # Create test file OUTSIDE fm_root to avoid duplication on import
    p = tmp_path / "delete_me.txt"
    p.write_text("delete this")
    display_name = local.import_file(p)
    assert local.exists(display_name)

    # Parse the file to add it to Unify logs so we can query for file_id
    local.parse(display_name)

    # Get the file_id
    rows = local._filter_files(filter=f"filename == '{display_name}'")
    assert rows
    file_id = rows[0].file_id

    # Delete via the global file manager
    before_state = {"files": local.list()}
    gfm._delete_file(filesystem="local", file_id=file_id)
    after_state = {"files": local.list()}

    # Verify deletion (no judge here as it's a direct call, not an LLM interpretation)
    assert display_name not in after_state["files"]


@pytest.mark.asyncio
async def test_global_fm_organize_delete_file(
    fm_root,
    file_manager,
    global_file_manager,
    tmp_path,
):
    pass

    local = file_manager
    gfm = global_file_manager

    # Create test file OUTSIDE fm_root to avoid duplication on import
    p = tmp_path / "delete_through_organize.txt"
    p.write_text("delete this via organize")
    filename = local.import_file(p)
    assert local.exists(filename)

    # Parse the file to add it to Unify logs so we can query for file_id
    local.parse(filename)

    # Get the file_id
    rows = local._filter_files(filter=f"filename == '{filename}'")
    assert rows
    file_id = rows[0].file_id

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
async def test_global_fm_search_and_filter_namespaced(global_file_manager):
    gfm = global_file_manager

    # Ensure search returns namespaced filenames and filter maintains source_filesystem
    results = gfm._filter_files()
    assert all(
        isinstance(r.get("filename", ""), str)
        and r.get("filename", "").startswith("/local/")
        for r in results
    )

    top = gfm._search_files(references={"full_text": "content"}, k=1)
    assert isinstance(top, list)
    if top:
        assert top[0]["filename"].startswith("/local/")
