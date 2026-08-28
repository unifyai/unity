"""FastAPI routes for the SharePoint channel.

Ports ``communication/sharepoint/views.py``. Only one translation
needed: ``from communication.helpers import get_graph_client`` ->
``from unify.gateway.common.graph import get_graph_client`` (the
helper was promoted to common in Phase B.4.prep). Everything else
is direct Microsoft Graph SDK usage that needs no rewrite.

Wire behaviour preserved bit-for-bit so the gateway aggregator can
mount this router at ``/sharepoint`` and external callers see no
change.
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, Response
from msgraph.generated.models.drive_item import DriveItem
from msgraph.generated.models.folder import Folder

from unify.gateway.common.graph import get_graph_client

logger = logging.getLogger("unify.gateway.channels.sharepoint")

router = APIRouter()


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------


@router.get("/sites")
async def list_sites(user_email: str, search: Optional[str] = None):
    """List SharePoint sites the user has access to."""
    try:
        graph = await get_graph_client(user_email)

        if search:
            sites = await graph.sites.get(
                request_configuration=lambda c: setattr(
                    c.query_parameters,
                    "search",
                    search,
                ),
            )
        else:
            sites = await graph.sites.get()

        return {
            "sites": [
                {
                    "id": site.id,
                    "name": site.display_name,
                    "web_url": site.web_url,
                    "description": site.description,
                }
                for site in (sites.value or [])
            ],
        }
    except Exception as exc:
        logger.error("failed to list sites: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sites/{site_id}")
async def get_site(user_email: str, site_id: str):
    """Get details of a specific SharePoint site."""
    try:
        graph = await get_graph_client(user_email)
        site = await graph.sites.by_site_id(site_id).get()

        return {
            "id": site.id,
            "name": site.display_name,
            "web_url": site.web_url,
            "description": site.description,
        }
    except Exception as exc:
        logger.error("failed to get site: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Drives (document libraries)
# ---------------------------------------------------------------------------


@router.get("/drives")
async def list_user_drives(
    user_email: Optional[str] = None,
    assistant_id: Optional[str] = None,
):
    """List user's OneDrive and accessible drives."""
    try:
        graph = await get_graph_client(user_email, assistant_id=assistant_id)

        my_drive = await graph.me.drive.get()
        drives = await graph.me.drives.get()

        all_drives = []
        if my_drive:
            all_drives.append(
                {
                    "id": my_drive.id,
                    "name": my_drive.name or "OneDrive",
                    "drive_type": my_drive.drive_type,
                    "web_url": my_drive.web_url,
                    "is_personal": True,
                },
            )

        for drive in drives.value or []:
            if drive.id != my_drive.id:
                all_drives.append(
                    {
                        "id": drive.id,
                        "name": drive.name,
                        "drive_type": drive.drive_type,
                        "web_url": drive.web_url,
                        "is_personal": False,
                    },
                )

        return {"drives": all_drives}
    except Exception as exc:
        logger.error("failed to list drives: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sites/{site_id}/drives")
async def list_site_drives(user_email: str, site_id: str):
    """List drives (document libraries) in a SharePoint site."""
    try:
        graph = await get_graph_client(user_email)
        drives = await graph.sites.by_site_id(site_id).drives.get()

        return {
            "drives": [
                {
                    "id": drive.id,
                    "name": drive.name,
                    "drive_type": drive.drive_type,
                    "web_url": drive.web_url,
                }
                for drive in (drives.value or [])
            ],
        }
    except Exception as exc:
        logger.error("failed to list site drives: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Files + folders
# ---------------------------------------------------------------------------


async def _drive_ref(graph, drive_id: str):
    """Resolve a drive to ``(drive_id, DriveItemRequestBuilder)``.

    The SDK's ``graph.me.drive`` (DriveRequestBuilder) only supports fetching the
    drive itself -- it exposes no item/root navigation -- so every item operation
    must go through ``graph.drives.by_drive_id(...)``. Resolve the personal
    drive's real id first when ``me`` is requested, then return the id alongside
    the builder (the id is needed to build path-addressed URLs below).
    """
    if drive_id == "me":
        me_drive = await graph.me.drive.get()
        drive_id = me_drive.id
    return drive_id, graph.drives.by_drive_id(drive_id)


def _path_item_url(graph, drive_id: str, path: str, suffix: str) -> str:
    """Absolute Graph URL for a path-addressed drive-item navigation.

    ``drive.root`` no longer exposes ``item_with_path`` in the SDK, so build the
    ``/drives/{id}/root:/{path}:/{suffix}`` URL explicitly and bind it onto the
    matching request builder via ``with_url``. Path separators stay literal so
    the colon-addressing hierarchy is preserved.
    """
    base = graph.request_adapter.base_url.rstrip("/")
    encoded = quote(path, safe="/")
    return f"{base}/drives/{drive_id}/root:/{encoded}:/{suffix}"


@router.get("/drives/{drive_id}/items")
async def list_items(
    drive_id: str,
    path: Optional[str] = None,
    item_id: Optional[str] = None,
    user_email: Optional[str] = None,
    assistant_id: Optional[str] = None,
):
    """List files and folders in a drive (by root, by path, or by item_id)."""
    try:
        graph = await get_graph_client(user_email, assistant_id=assistant_id)
        drive_id, drive = await _drive_ref(graph, drive_id)

        if item_id:
            items = await drive.items.by_drive_item_id(item_id).children.get()
        elif path:
            items = (
                await drive.items.by_drive_item_id("root")
                .children.with_url(_path_item_url(graph, drive_id, path, "children"))
                .get()
            )
        else:
            # The root's children are reached via the drive-item builder using
            # Graph's reserved ``root`` item id -- ``drive.root`` has no
            # ``children`` navigation in the SDK.
            items = await drive.items.by_drive_item_id("root").children.get()

        return {
            "items": [
                {
                    "id": item.id,
                    "name": item.name,
                    "type": "folder" if item.folder else "file",
                    "size": item.size,
                    "mime_type": item.file.mime_type if item.file else None,
                    "created": (
                        item.created_date_time.isoformat()
                        if item.created_date_time
                        else None
                    ),
                    "modified": (
                        item.last_modified_date_time.isoformat()
                        if item.last_modified_date_time
                        else None
                    ),
                    "web_url": item.web_url,
                }
                for item in (items.value or [])
            ],
        }
    except Exception as exc:
        logger.error("failed to list items: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/drives/{drive_id}/items/{item_id}")
async def get_item(user_email: str, drive_id: str, item_id: str):
    """Get metadata for a specific file or folder."""
    try:
        graph = await get_graph_client(user_email)
        _, drive = await _drive_ref(graph, drive_id)

        item = await drive.items.by_drive_item_id(item_id).get()

        return {
            "id": item.id,
            "name": item.name,
            "type": "folder" if item.folder else "file",
            "size": item.size,
            "mime_type": item.file.mime_type if item.file else None,
            "created": (
                item.created_date_time.isoformat() if item.created_date_time else None
            ),
            "modified": (
                item.last_modified_date_time.isoformat()
                if item.last_modified_date_time
                else None
            ),
            "web_url": item.web_url,
            "parent_path": (
                item.parent_reference.path if item.parent_reference else None
            ),
        }
    except Exception as exc:
        logger.error("failed to get item: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/drives/{drive_id}/items/{item_id}/content")
async def download_file(user_email: str, drive_id: str, item_id: str):
    """Download a file's content."""
    try:
        graph = await get_graph_client(user_email)
        _, drive = await _drive_ref(graph, drive_id)

        item = await drive.items.by_drive_item_id(item_id).get()
        if item.folder:
            raise HTTPException(status_code=400, detail="Cannot download a folder")

        content = await drive.items.by_drive_item_id(item_id).content.get()

        return Response(
            content=content,
            media_type=(
                item.file.mime_type if item.file else "application/octet-stream"
            ),
            headers={"Content-Disposition": f"attachment; filename={item.name}"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("failed to download file: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.put("/drives/{drive_id}/upload")
async def upload_file(request: Request, user_email: str, drive_id: str):
    """Upload a file to a drive.

    Request body: ``{"path": "folder/file.txt", "content": "<b64>", ...}``.
    Files > 4MB require chunked upload (not implemented here).
    """
    try:
        data = await request.json()
        path = data.get("path")
        content = data.get("content")

        if not path or content is None:
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: path, content",
            )

        try:
            file_bytes = base64.b64decode(content, validate=True)
        except (binascii.Error, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="content must be valid base64",
            ) from exc

        graph = await get_graph_client(user_email)
        drive_id, drive = await _drive_ref(graph, drive_id)

        result = (
            await drive.items.by_drive_item_id("root")
            .content.with_url(_path_item_url(graph, drive_id, path, "content"))
            .put(file_bytes)
        )

        return {
            "success": True,
            "id": result.id,
            "name": result.name,
            "web_url": result.web_url,
            "size": result.size,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("failed to upload file: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/drives/{drive_id}/folder")
async def create_folder(request: Request, user_email: str, drive_id: str):
    """Create a folder."""
    try:
        data = await request.json()
        folder_name = data.get("name")
        parent_path = data.get("parent_path")

        if not folder_name:
            raise HTTPException(status_code=400, detail="Missing required field: name")

        graph = await get_graph_client(user_email)
        drive_id, drive = await _drive_ref(graph, drive_id)

        new_folder = DriveItem(name=folder_name, folder=Folder())

        if parent_path:
            result = (
                await drive.items.by_drive_item_id("root")
                .children.with_url(
                    _path_item_url(graph, drive_id, parent_path, "children"),
                )
                .post(new_folder)
            )
        else:
            result = await drive.items.by_drive_item_id("root").children.post(
                new_folder,
            )

        return {
            "success": True,
            "id": result.id,
            "name": result.name,
            "web_url": result.web_url,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("failed to create folder: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/drives/{drive_id}/items/{item_id}")
async def delete_item(user_email: str, drive_id: str, item_id: str):
    """Delete a file or folder."""
    try:
        graph = await get_graph_client(user_email)
        _, drive = await _drive_ref(graph, drive_id)

        await drive.items.by_drive_item_id(item_id).delete()

        return {"success": True}
    except Exception as exc:
        logger.error("failed to delete item: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@router.get("/drives/{drive_id}/search")
async def search_files(user_email: str, drive_id: str, q: str):
    """Search for files in a drive."""
    try:
        graph = await get_graph_client(user_email)
        _, drive = await _drive_ref(graph, drive_id)

        # ``search_with_q`` is a navigation on the drive builder, not on
        # ``drive.root``.
        results = await drive.search_with_q(q).get()

        return {
            "results": [
                {
                    "id": item.id,
                    "name": item.name,
                    "type": "folder" if item.folder else "file",
                    "path": (
                        item.parent_reference.path if item.parent_reference else None
                    ),
                    "web_url": item.web_url,
                }
                for item in (results.value or [])
            ],
        }
    except Exception as exc:
        logger.error("failed to search files: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


__all__ = ["router"]
