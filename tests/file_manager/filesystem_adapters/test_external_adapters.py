from __future__ import annotations

import pytest

from unity.file_manager.filesystem_adapters.codesandbox_adapter import (
    CodeSandboxFileSystemAdapter,
)
from unity.file_manager.filesystem_adapters.interact_adapter import (
    InteractFileSystemAdapter,
)


@pytest.mark.asyncio
async def test_codesandbox_http_calls(csb_http):
    # Configure mocked HTTP responses
    csb_http.set_post_status(200)
    csb_http.set_get_json({"items": ["a.txt", "b.txt"]}, status=200)

    ad = CodeSandboxFileSystemAdapter(
        "sbx-123",
        auth_token="tkn",
        client=None,
        service_base_url="http://svc",
    )

    files = list(ad.iter_files("/"))
    names = [f.name for f in files]
    assert set(names) == {"a.txt", "b.txt"}

    # First call should POST to connect and GET readdir
    assert any("/sandboxes/sbx-123/connect" in c["url"] for c in csb_http.post_calls)
    assert any("/fs/sbx-123/readdir" in c["url"] for c in csb_http.get_calls)

    # Test open_bytes fallback
    content = b"hello"
    csb_http.set_get_json({}, status=200, content=content)
    out = ad.open_bytes("/a.txt")
    assert out == content

    # Test rename error path
    csb_http.set_post_status(500)
    with pytest.raises(RuntimeError):
        ad.rename("/a.txt", "z.txt")


@pytest.mark.asyncio
async def test_interact_token_and_search(interact_urlopen, monkeypatch):
    # Set environment variables for test (adapter requires some env vars like person_id)
    monkeypatch.setenv("UNITY_FILE_INTERACT_API_BASE", "https://api.example")
    monkeypatch.setenv("UNITY_FILE_INTERACT_KEY", "test_key")
    monkeypatch.setenv("UNITY_FILE_INTERACT_SECRET", "test_secret")
    monkeypatch.setenv("UNITY_FILE_INTERACT_PERSON_ID", "123")
    monkeypatch.setenv("UNITY_FILE_INTERACT_TENANT", "tenant123")

    # Configure token and search responses (use Interact API's actual field names)
    interact_urlopen.set(
        "/token?personid=",
        {"access_token": "abc", "expires_in": 1200},
    )
    interact_urlopen.set(
        "/api/search",
        {"Results": [{"Id": "1", "Title": "doc1"}, {"Id": "2", "Title": "doc2"}]},
    )
    interact_urlopen.set(
        "/api/resource/stream",
        {"content": "aGVsbG8="},
    )  # base64 hello

    # Constructor args are ignored when env vars are set
    ad = InteractFileSystemAdapter(
        api_base="https://fallback.example",
        api_key="fallback_key",
        space="default",
    )

    files = list(ad.iter_files("/"))
    assert len(files) == 2
    assert files[0].name in {"doc1", "doc2"}  # Title field
    assert files[0].path in {"/1", "/2"}  # Path reflects the ID

    # get_file returns a reference with metadata where possible
    ref = ad.get_file("1")
    assert ref.path == "/1"
    assert ref.name in {"doc1", "1"}

    # open_bytes returns decoded bytes
    data = ad.open_bytes("1")
    assert data == b"hello"

    # rename/move not implemented
    with pytest.raises(NotImplementedError):
        ad.rename("1", "new")
    with pytest.raises(NotImplementedError):
        ad.move("1", "/x")
