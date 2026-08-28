"""Behavioural tests for ``unify.gateway.channels.sharepoint``.

11 endpoints; tests cover router contract + happy-path per endpoint
+ key edge cases (drive_id = "me" vs explicit id, missing required
fields). No existing tests in ``communication/tests/sharepoint/`` to
port.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from unify.gateway.channels.sharepoint import router

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/sharepoint")
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _make_graph_mock(me_drive_id: str = "me-drive-id") -> MagicMock:
    """Graph client mock matching the real SDK's navigation surface.

    ``graph.me.drive`` (DriveRequestBuilder) only supports ``get`` -- it has no
    item/root navigation -- and ``drive.root`` (RootRequestBuilder) only supports
    ``get``/``content``. Restricting both with ``spec`` means code that reverts to
    the removed navigations (``me.drive.items``, ``drive.root.children``,
    ``drive.root.item_with_path``, ``drive.root.search_with_q``) raises here rather
    than only blowing up against real Graph. ``request_adapter.base_url`` is a real
    string so path-addressed ``with_url`` targets can be asserted exactly.
    """
    graph = MagicMock()
    graph.request_adapter.base_url = "https://graph.microsoft.com/v1.0"
    graph.me.drive = MagicMock(spec=["get"])
    graph.me.drive.get = AsyncMock(return_value=MagicMock(id=me_drive_id))
    graph.drives.by_drive_id.return_value.root = MagicMock(spec=["get", "content"])
    return graph


# ---------------------------------------------------------------------------
# Router contract
# ---------------------------------------------------------------------------


def test_router_exposes_expected_paths() -> None:
    paths = sorted(
        (r.path, sorted(r.methods)) for r in router.routes  # type: ignore[attr-defined]
    )
    assert paths == [
        ("/drives", ["GET"]),
        ("/drives/{drive_id}/folder", ["POST"]),
        ("/drives/{drive_id}/items", ["GET"]),
        ("/drives/{drive_id}/items/{item_id}", ["DELETE"]),
        ("/drives/{drive_id}/items/{item_id}", ["GET"]),
        ("/drives/{drive_id}/items/{item_id}/content", ["GET"]),
        ("/drives/{drive_id}/search", ["GET"]),
        ("/drives/{drive_id}/upload", ["PUT"]),
        ("/sites", ["GET"]),
        ("/sites/{site_id}", ["GET"]),
        ("/sites/{site_id}/drives", ["GET"]),
    ]


def test_router_importable_from_package_root() -> None:
    from unify.gateway.channels.sharepoint import router as exported

    assert exported is router


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


def test_list_sites_returns_site_records(client: TestClient) -> None:
    site = MagicMock(
        id="site-1",
        display_name="Site A",
        web_url="https://example.com/site-a",
        description="...",
    )
    fake_graph = _make_graph_mock()
    fake_graph.sites.get = AsyncMock(return_value=MagicMock(value=[site]))
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.get("/sharepoint/sites", params={"user_email": "u@x.com"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sites"]) == 1
    assert body["sites"][0]["id"] == "site-1"
    assert body["sites"][0]["name"] == "Site A"


def test_get_site_returns_single_site(client: TestClient) -> None:
    site = MagicMock(id="site-1", display_name="Site A", web_url="x", description="d")
    fake_graph = _make_graph_mock()
    fake_graph.sites.by_site_id.return_value.get = AsyncMock(return_value=site)
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.get(
            "/sharepoint/sites/site-1",
            params={"user_email": "u@x.com"},
        )
    assert resp.status_code == 200
    assert resp.json()["id"] == "site-1"


# ---------------------------------------------------------------------------
# Drives
# ---------------------------------------------------------------------------


def test_list_user_drives_marks_personal_drive(client: TestClient) -> None:
    """Personal drive is_personal=True; others False."""
    my_drive = MagicMock(
        id="onedrive-1",
        name="My OneDrive",
        drive_type="personal",
        web_url="https://example.com/me",
    )
    other = MagicMock(
        id="other-1",
        name="Other",
        drive_type="business",
        web_url="https://example.com/other",
    )
    fake_graph = _make_graph_mock()
    fake_graph.me.drive.get = AsyncMock(return_value=my_drive)
    fake_graph.me.drives.get = AsyncMock(
        return_value=MagicMock(value=[my_drive, other]),
    )
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.get("/sharepoint/drives", params={"user_email": "u@x.com"})
    assert resp.status_code == 200
    drives = resp.json()["drives"]
    assert len(drives) == 2
    assert drives[0]["is_personal"] is True
    assert drives[1]["is_personal"] is False


def test_list_site_drives(client: TestClient) -> None:
    drive = MagicMock(
        id="d-1",
        name="Docs",
        drive_type="documentLibrary",
        web_url="x",
    )
    fake_graph = _make_graph_mock()
    fake_graph.sites.by_site_id.return_value.drives.get = AsyncMock(
        return_value=MagicMock(value=[drive]),
    )
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.get(
            "/sharepoint/sites/site-1/drives",
            params={"user_email": "u@x.com"},
        )
    assert resp.status_code == 200
    assert resp.json()["drives"][0]["id"] == "d-1"


# ---------------------------------------------------------------------------
# Items (list / get / download)
# ---------------------------------------------------------------------------


def _mock_drive_item(
    *,
    id_: str = "item-1",
    name: str = "file.txt",
    is_folder: bool = False,
) -> MagicMock:
    item = MagicMock()
    item.id = id_
    item.name = name
    item.folder = MagicMock() if is_folder else None
    item.file = MagicMock(mime_type="text/plain") if not is_folder else None
    item.size = 42
    item.created_date_time = None
    item.last_modified_date_time = None
    item.web_url = "https://example.com/file"
    item.parent_reference = MagicMock(path="/Documents")
    return item


def test_list_items_root_default(client: TestClient) -> None:
    """No path / no item_id -> list root children via the reserved ``root`` id.

    Also exercises ``me`` resolution: ``graph.me.drive`` can't navigate items, so
    the endpoint resolves the personal drive id and goes through
    ``graph.drives.by_drive_id(...)``.
    """
    item = _mock_drive_item()
    fake_graph = _make_graph_mock(me_drive_id="me-drive-id")
    drive = fake_graph.drives.by_drive_id.return_value
    drive.items.by_drive_item_id.return_value.children.get = AsyncMock(
        return_value=MagicMock(value=[item]),
    )
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.get(
            "/sharepoint/drives/me/items",
            params={"user_email": "u@x.com"},
        )
    assert resp.status_code == 200
    assert resp.json()["items"][0]["type"] == "file"
    fake_graph.me.drive.get.assert_awaited()  # me resolved to a real drive id
    fake_graph.drives.by_drive_id.assert_called_with("me-drive-id")
    drive.items.by_drive_item_id.assert_called_with("root")


def test_list_items_explicit_drive_id_uses_drives_by_id(
    client: TestClient,
) -> None:
    """drive_id != 'me' -> graph.drives.by_drive_id directly (no me-resolution)."""
    item = _mock_drive_item(name="other.txt")
    fake_graph = _make_graph_mock()
    drive = fake_graph.drives.by_drive_id.return_value
    drive.items.by_drive_item_id.return_value.children.get = AsyncMock(
        return_value=MagicMock(value=[item]),
    )
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.get(
            "/sharepoint/drives/external-drive-1/items",
            params={"user_email": "u@x.com"},
        )
    assert resp.status_code == 200
    fake_graph.me.drive.get.assert_not_awaited()
    fake_graph.drives.by_drive_id.assert_called_with("external-drive-1")
    drive.items.by_drive_item_id.assert_called_with("root")


def test_list_items_by_path(client: TestClient) -> None:
    """A ``path`` lists children of the path-addressed folder via ``with_url``.

    ``drive.root.item_with_path`` no longer exists, so children are listed by
    binding the ``/root:/{path}:/children`` URL onto the children builder.
    """
    item = _mock_drive_item(is_folder=True, name="Reports")
    fake_graph = _make_graph_mock(me_drive_id="me-drive-id")
    drive = fake_graph.drives.by_drive_id.return_value
    children = drive.items.by_drive_item_id.return_value.children
    children.with_url.return_value.get = AsyncMock(
        return_value=MagicMock(value=[item]),
    )
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.get(
            "/sharepoint/drives/me/items",
            params={"user_email": "u@x.com", "path": "Documents/Reports"},
        )
    assert resp.status_code == 200
    assert resp.json()["items"][0]["type"] == "folder"
    children.with_url.assert_called_with(
        "https://graph.microsoft.com/v1.0/drives/me-drive-id"
        "/root:/Documents/Reports:/children",
    )


def test_get_item(client: TestClient) -> None:
    item = _mock_drive_item(id_="x-1")
    fake_graph = _make_graph_mock()
    drive = fake_graph.drives.by_drive_id.return_value
    drive.items.by_drive_item_id.return_value.get = AsyncMock(return_value=item)
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.get(
            "/sharepoint/drives/me/items/x-1",
            params={"user_email": "u@x.com"},
        )
    assert resp.status_code == 200
    assert resp.json()["id"] == "x-1"
    drive.items.by_drive_item_id.assert_called_with("x-1")


def test_download_folder_returns_400(client: TestClient) -> None:
    item = _mock_drive_item(is_folder=True)
    fake_graph = _make_graph_mock()
    drive = fake_graph.drives.by_drive_id.return_value
    drive.items.by_drive_item_id.return_value.get = AsyncMock(return_value=item)
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.get(
            "/sharepoint/drives/me/items/folder-1/content",
            params={"user_email": "u@x.com"},
        )
    assert resp.status_code == 400


def test_download_file_returns_bytes_with_content_type(client: TestClient) -> None:
    item = _mock_drive_item()
    fake_graph = _make_graph_mock()
    drive = fake_graph.drives.by_drive_id.return_value
    drive.items.by_drive_item_id.return_value.get = AsyncMock(return_value=item)
    drive.items.by_drive_item_id.return_value.content.get = AsyncMock(
        return_value=b"file-content",
    )
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.get(
            "/sharepoint/drives/me/items/file-1/content",
            params={"user_email": "u@x.com"},
        )
    assert resp.status_code == 200
    assert resp.content == b"file-content"
    assert "text/plain" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# Upload / create folder / delete
# ---------------------------------------------------------------------------


def test_upload_missing_required_fields_returns_400(client: TestClient) -> None:
    resp = client.put(
        "/sharepoint/drives/me/upload",
        params={"user_email": "u@x.com"},
        json={"path": "x"},  # missing content
    )
    assert resp.status_code == 400


def test_upload_decodes_base64_content(client: TestClient) -> None:
    """The endpoint base64-decodes the supplied content."""
    file_bytes = b"hello world"
    content_b64 = base64.b64encode(file_bytes).decode("utf-8")

    fake_graph = _make_graph_mock(me_drive_id="me-drive-id")
    uploaded = MagicMock(id="u-1", name="x.txt", web_url="x", size=11)
    drive = fake_graph.drives.by_drive_id.return_value
    content = drive.items.by_drive_item_id.return_value.content

    captured: dict = {}

    async def _capture(payload):
        captured["bytes"] = payload
        return uploaded

    content.with_url.return_value.put = _capture

    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.put(
            "/sharepoint/drives/me/upload",
            params={"user_email": "u@x.com"},
            json={"path": "x.txt", "content": content_b64},
        )

    assert resp.status_code == 200
    assert captured["bytes"] == file_bytes
    content.with_url.assert_called_with(
        "https://graph.microsoft.com/v1.0/drives/me-drive-id/root:/x.txt:/content",
    )


@pytest.mark.parametrize(
    "content",
    [
        "not-real-base64-content!!",
        {"unexpected": "object"},
    ],
)
def test_upload_rejects_invalid_base64_before_graph_lookup(
    client: TestClient,
    content: object,
) -> None:
    get_graph_client = AsyncMock()
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=get_graph_client,
    ):
        resp = client.put(
            "/sharepoint/drives/me/upload",
            params={"user_email": "u@x.com"},
            json={"path": "x.txt", "content": content},
        )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "content must be valid base64"
    get_graph_client.assert_not_awaited()


def test_upload_accepts_empty_base64_content(client: TestClient) -> None:
    fake_graph = _make_graph_mock()
    uploaded = MagicMock(id="u-1", name="empty.txt", web_url="x", size=0)
    captured: dict = {}

    async def _capture(payload):
        captured["bytes"] = payload
        return uploaded

    content = (
        fake_graph.drives.by_drive_id.return_value.items.by_drive_item_id.return_value.content
    )  # noqa: E501
    content.with_url.return_value.put = _capture

    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.put(
            "/sharepoint/drives/me/upload",
            params={"user_email": "u@x.com"},
            json={"path": "empty.txt", "content": ""},
        )

    assert resp.status_code == 200
    assert captured["bytes"] == b""


def test_create_folder_missing_name_returns_400(client: TestClient) -> None:
    resp = client.post(
        "/sharepoint/drives/me/folder",
        params={"user_email": "u@x.com"},
        json={},
    )
    assert resp.status_code == 400


def test_create_folder_at_root(client: TestClient) -> None:
    fake_graph = _make_graph_mock()
    new_folder = MagicMock(id="f-1", name="New", web_url="x")
    drive = fake_graph.drives.by_drive_id.return_value
    drive.items.by_drive_item_id.return_value.children.post = AsyncMock(
        return_value=new_folder,
    )
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.post(
            "/sharepoint/drives/me/folder",
            params={"user_email": "u@x.com"},
            json={"name": "New"},
        )
    assert resp.status_code == 200
    assert resp.json()["id"] == "f-1"
    drive.items.by_drive_item_id.assert_called_with("root")


def test_create_folder_with_parent_path(client: TestClient) -> None:
    fake_graph = _make_graph_mock(me_drive_id="me-drive-id")
    new_folder = MagicMock(id="f-1", name="Sub", web_url="x")
    drive = fake_graph.drives.by_drive_id.return_value
    children = drive.items.by_drive_item_id.return_value.children
    children.with_url.return_value.post = AsyncMock(return_value=new_folder)
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.post(
            "/sharepoint/drives/me/folder",
            params={"user_email": "u@x.com"},
            json={"name": "Sub", "parent_path": "Documents/Project"},
        )
    assert resp.status_code == 200
    children.with_url.assert_called_with(
        "https://graph.microsoft.com/v1.0/drives/me-drive-id"
        "/root:/Documents/Project:/children",
    )


def test_delete_item_me_drive(client: TestClient) -> None:
    fake_graph = _make_graph_mock(me_drive_id="me-drive-id")
    drive = fake_graph.drives.by_drive_id.return_value
    drive.items.by_drive_item_id.return_value.delete = AsyncMock()
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.request(
            "DELETE",
            "/sharepoint/drives/me/items/x-1",
            params={"user_email": "u@x.com"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"success": True}
    fake_graph.me.drive.get.assert_awaited()
    fake_graph.drives.by_drive_id.assert_called_with("me-drive-id")


def test_delete_item_explicit_drive(client: TestClient) -> None:
    fake_graph = _make_graph_mock()
    fake_graph.drives.by_drive_id.return_value.items.by_drive_item_id.return_value.delete = (
        AsyncMock()
    )  # noqa: E501
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.request(
            "DELETE",
            "/sharepoint/drives/external/items/x-1",
            params={"user_email": "u@x.com"},
        )
    assert resp.status_code == 200
    fake_graph.me.drive.get.assert_not_awaited()
    fake_graph.drives.by_drive_id.assert_called_with("external")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_returns_matching_items(client: TestClient) -> None:
    item = _mock_drive_item(name="quarterly.pdf")
    fake_graph = _make_graph_mock()
    drive = fake_graph.drives.by_drive_id.return_value
    drive.search_with_q.return_value.get = AsyncMock(
        return_value=MagicMock(value=[item]),
    )
    with patch(
        "unify.gateway.channels.sharepoint.views.get_graph_client",
        new=AsyncMock(return_value=fake_graph),
    ):
        resp = client.get(
            "/sharepoint/drives/me/search",
            params={"user_email": "u@x.com", "q": "quarterly"},
        )
    assert resp.status_code == 200
    drive.search_with_q.assert_called_with("quarterly")
    assert resp.json()["results"][0]["name"] == "quarterly.pdf"
