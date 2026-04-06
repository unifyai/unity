"""
Concrete DashboardManager implementation.

Delegates tile/dashboard CRUD to DataManager for Unify context operations
and to token_ops for Orchestra token registration. Stays thin.

Docstrings are inherited from BaseDashboardManager via @functools.wraps.
"""

from __future__ import annotations

import functools
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from unity.common.context_registry import ContextRegistry, TableContext
from unity.common.model_to_fields import model_to_fields
from unity.dashboard_manager.base import BaseDashboardManager
from unity.dashboard_manager.ops.dashboard_ops import (
    build_dashboard_record_row,
    deserialize_layout,
    serialize_layout,
)
from unity.dashboard_manager.ops.tile_ops import (
    build_tile_record_row,
    validate_data_bindings,
    verify_data_bindings,
)
from unity.dashboard_manager.ops.token_ops import (
    delete_token,
    generate_token,
    register_token,
)
from unity.dashboard_manager.types.dashboard import (
    DashboardRecord,
    DashboardRecordRow,
    DashboardResult,
    TilePosition,
)
from unity.dashboard_manager.types.tile import (
    DataBinding,
    TileRecord,
    TileRecordRow,
    TileResult,
)
from unity.settings import SETTINGS

logger = logging.getLogger(__name__)


def _get_active_project() -> str:
    """Get the currently active Unify project name."""
    try:
        import unify

        project = unify.active_project()
        return project or ""
    except Exception:
        return ""


def _build_tile_url(token: str) -> str:
    console_url = getattr(SETTINGS, "CONSOLE_URL", "https://console.unify.ai")
    if isinstance(console_url, str):
        console_url = console_url.rstrip("/")
    else:
        console_url = "https://console.unify.ai"
    return f"{console_url}/tile/view/{token}"


def _build_dashboard_url(token: str) -> str:
    console_url = getattr(SETTINGS, "CONSOLE_URL", "https://console.unify.ai")
    if isinstance(console_url, str):
        console_url = console_url.rstrip("/")
    else:
        console_url = "https://console.unify.ai"
    return f"{console_url}/dashboard/view/{token}"


class DashboardManager(BaseDashboardManager):
    """Concrete DashboardManager backed by Unify contexts and Orchestra tokens."""

    class Config:
        """Context registration for DashboardManager's owned namespace."""

        required_contexts = [
            TableContext(
                name="Dashboards/Tiles",
                description=(
                    "Registry of dashboard tiles (HTML visualization artifacts). "
                    "Each row stores tile metadata and the full HTML content."
                ),
                fields=model_to_fields(TileRecordRow),
                unique_keys={"tile_id": "int"},
                auto_counting={"tile_id": None},
            ),
            TableContext(
                name="Dashboards/Layouts",
                description=(
                    "Registry of composed dashboard layouts. "
                    "Each row stores layout metadata and tile positions as JSON."
                ),
                fields=model_to_fields(DashboardRecordRow),
                unique_keys={"dashboard_id": "int"},
                auto_counting={"dashboard_id": None},
            ),
        ]

    def __init__(self) -> None:
        super().__init__()
        self.include_in_multi_assistant_table = True
        try:
            self._tiles_ctx = ContextRegistry.get_context(self, "Dashboards/Tiles")
        except Exception:
            self._tiles_ctx = "Dashboards/Tiles"
        try:
            self._layouts_ctx = ContextRegistry.get_context(
                self,
                "Dashboards/Layouts",
            )
        except Exception:
            self._layouts_ctx = "Dashboards/Layouts"
        logger.debug(
            "DashboardManager initialized: tiles=%s layouts=%s",
            self._tiles_ctx,
            self._layouts_ctx,
        )

    def _get_dm(self):
        from unity.manager_registry import ManagerRegistry

        return ManagerRegistry.get_data_manager()

    # ──────────────────────────────────────────────────────────────────────────
    # Tiles
    # ──────────────────────────────────────────────────────────────────────────

    @functools.wraps(BaseDashboardManager.create_tile, updated=())
    def create_tile(
        self,
        html: str,
        *,
        title: str,
        description: Optional[str] = None,
        data_bindings: Optional[List[DataBinding]] = None,
    ) -> TileResult:
        try:
            token = generate_token()
            bindings = validate_data_bindings(data_bindings)

            if bindings:
                verify_data_bindings(bindings, self._get_dm())

            row = build_tile_record_row(
                token=token,
                html=html,
                title=title,
                description=description,
                data_bindings=bindings,
            )

            dm = self._get_dm()
            dm.insert_rows(self._tiles_ctx, [row.model_dump()])

            register_token(token, "tile", self._tiles_ctx, _get_active_project())

            return TileResult(
                url=_build_tile_url(token),
                token=token,
                title=title,
            )
        except Exception as e:
            logger.exception("create_tile failed")
            return TileResult(error=str(e))

    @functools.wraps(BaseDashboardManager.get_tile, updated=())
    def get_tile(self, token: str) -> Optional[TileRecord]:
        dm = self._get_dm()
        rows = dm.filter(
            self._tiles_ctx,
            filter=f"token == '{token}'",
            limit=1,
        )
        if not rows:
            return None
        return TileRecord(**rows[0])

    @functools.wraps(BaseDashboardManager.update_tile, updated=())
    def update_tile(
        self,
        token: str,
        *,
        html: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> TileResult:
        try:
            updates: Dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if html is not None:
                updates["html_content"] = html
            if title is not None:
                updates["title"] = title
            if description is not None:
                updates["description"] = description

            dm = self._get_dm()
            dm.update_rows(
                self._tiles_ctx,
                updates,
                filter=f"token == '{token}'",
            )

            return TileResult(
                url=_build_tile_url(token),
                token=token,
                title=title,
            )
        except Exception as e:
            logger.exception("update_tile failed")
            return TileResult(error=str(e))

    @functools.wraps(BaseDashboardManager.delete_tile, updated=())
    def delete_tile(self, token: str) -> bool:
        dm = self._get_dm()
        deleted = dm.delete_rows(
            self._tiles_ctx,
            filter=f"token == '{token}'",
        )
        if deleted:
            delete_token(token)
        return deleted > 0

    @functools.wraps(BaseDashboardManager.list_tiles, updated=())
    def list_tiles(
        self,
        *,
        filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[TileRecord]:
        dm = self._get_dm()
        rows = dm.filter(
            self._tiles_ctx,
            filter=filter,
            exclude_columns=["html_content"],
            limit=limit,
        )
        result = []
        for row in rows:
            row.setdefault("html_content", "")
            result.append(TileRecord(**row))
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Dashboards
    # ──────────────────────────────────────────────────────────────────────────

    @functools.wraps(BaseDashboardManager.create_dashboard, updated=())
    def create_dashboard(
        self,
        title: str,
        *,
        description: Optional[str] = None,
        tiles: Optional[List[TilePosition]] = None,
    ) -> DashboardResult:
        try:
            token = generate_token()
            tile_list = tiles or []
            row = build_dashboard_record_row(
                token=token,
                title=title,
                tiles=tile_list,
                description=description,
            )

            dm = self._get_dm()
            dm.insert_rows(self._layouts_ctx, [row.model_dump()])

            register_token(token, "dashboard", self._layouts_ctx, _get_active_project())

            return DashboardResult(
                url=_build_dashboard_url(token),
                token=token,
                title=title,
                tiles=tile_list,
            )
        except Exception as e:
            logger.exception("create_dashboard failed")
            return DashboardResult(error=str(e))

    @functools.wraps(BaseDashboardManager.get_dashboard, updated=())
    def get_dashboard(self, token: str) -> Optional[DashboardResult]:
        dm = self._get_dm()
        rows = dm.filter(
            self._layouts_ctx,
            filter=f"token == '{token}'",
            limit=1,
        )
        if not rows:
            return None
        record = rows[0]
        tile_positions = deserialize_layout(record.get("layout", "[]"))
        return DashboardResult(
            url=_build_dashboard_url(token),
            token=token,
            title=record.get("title"),
            tiles=tile_positions,
        )

    @functools.wraps(BaseDashboardManager.update_dashboard, updated=())
    def update_dashboard(
        self,
        token: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        tiles: Optional[List[TilePosition]] = None,
    ) -> DashboardResult:
        try:
            updates: Dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if title is not None:
                updates["title"] = title
            if description is not None:
                updates["description"] = description
            if tiles is not None:
                updates["layout"] = serialize_layout(tiles)
                updates["tile_count"] = len(tiles)

            dm = self._get_dm()
            dm.update_rows(
                self._layouts_ctx,
                updates,
                filter=f"token == '{token}'",
            )

            final_tiles = tiles
            if final_tiles is None:
                existing = self.get_dashboard(token)
                final_tiles = existing.tiles if existing else []

            return DashboardResult(
                url=_build_dashboard_url(token),
                token=token,
                title=title,
                tiles=final_tiles,
            )
        except Exception as e:
            logger.exception("update_dashboard failed")
            return DashboardResult(error=str(e))

    @functools.wraps(BaseDashboardManager.delete_dashboard, updated=())
    def delete_dashboard(self, token: str) -> bool:
        dm = self._get_dm()
        deleted = dm.delete_rows(
            self._layouts_ctx,
            filter=f"token == '{token}'",
        )
        if deleted:
            delete_token(token)
        return deleted > 0

    @functools.wraps(BaseDashboardManager.list_dashboards, updated=())
    def list_dashboards(
        self,
        *,
        filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[DashboardRecord]:
        dm = self._get_dm()
        rows = dm.filter(
            self._layouts_ctx,
            filter=filter,
            limit=limit,
        )
        return [DashboardRecord(**row) for row in rows]
