"""
Simulated DashboardManager for testing.

In-memory implementation that mirrors the concrete DashboardManager's
behavior without requiring Orchestra or Unify backends. Uses dict-based
storage keyed by token.
"""

from __future__ import annotations

import functools
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from unity.dashboard_manager.base import BaseDashboardManager
from unity.dashboard_manager.ops.dashboard_ops import (
    serialize_layout,
    deserialize_layout,
)
from unity.dashboard_manager.types.dashboard import (
    DashboardRecord,
    DashboardResult,
    TilePosition,
)
from unity.dashboard_manager.ops.tile_ops import _contexts_for_binding
from unity.dashboard_manager.types.tile import (
    DataBinding,
    TileRecord,
    TileResult,
)


class SimulatedDashboardManager(BaseDashboardManager):
    """
    In-memory DashboardManager for testing and development.

    Stores tiles and dashboards in plain dictionaries. Does not contact
    Orchestra or Unify -- all operations are instant and deterministic.

    Usage
    -----
    >>> dm = SimulatedDashboardManager()
    >>> result = dm.create_tile("<h1>Hello</h1>", title="Test")
    >>> assert result.succeeded
    >>> tile = dm.get_tile(result.token)
    >>> assert tile.html_content == "<h1>Hello</h1>"
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._tiles: Dict[str, Dict[str, Any]] = {}
        self._dashboards: Dict[str, Dict[str, Any]] = {}
        self._tile_counter = 0
        self._dashboard_counter = 0

    def clear(self) -> None:
        """Reset all in-memory storage."""
        self._tiles.clear()
        self._dashboards.clear()
        self._tile_counter = 0
        self._dashboard_counter = 0

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
        self._tile_counter += 1
        token = f"sim_tile_{self._tile_counter:04d}"
        has_bindings = bool(data_bindings)
        binding_contexts = None
        if data_bindings:
            all_ctxs: list[str] = []
            for b in data_bindings:
                all_ctxs.extend(_contexts_for_binding(b))
            binding_contexts = ",".join(dict.fromkeys(all_ctxs))
        now = datetime.now(timezone.utc).isoformat()

        self._tiles[token] = {
            "tile_id": self._tile_counter,
            "token": token,
            "title": title,
            "description": description,
            "html_content": html,
            "has_data_bindings": has_bindings,
            "data_binding_contexts": binding_contexts,
            "created_at": now,
            "updated_at": now,
        }

        return TileResult(
            url=f"https://simulated-console.example.com/tile/view/{token}",
            token=token,
            title=title,
        )

    @functools.wraps(BaseDashboardManager.get_tile, updated=())
    def get_tile(self, token: str) -> Optional[TileRecord]:
        data = self._tiles.get(token)
        if data is None:
            return None
        return TileRecord(**data)

    @functools.wraps(BaseDashboardManager.update_tile, updated=())
    def update_tile(
        self,
        token: str,
        *,
        html: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> TileResult:
        if token not in self._tiles:
            return TileResult(error=f"Tile '{token}' not found")

        data = self._tiles[token]
        if html is not None:
            data["html_content"] = html
        if title is not None:
            data["title"] = title
        if description is not None:
            data["description"] = description
        data["updated_at"] = datetime.now(timezone.utc).isoformat()

        return TileResult(
            url=f"https://simulated-console.example.com/tile/view/{token}",
            token=token,
            title=data["title"],
        )

    @functools.wraps(BaseDashboardManager.delete_tile, updated=())
    def delete_tile(self, token: str) -> bool:
        return self._tiles.pop(token, None) is not None

    @functools.wraps(BaseDashboardManager.list_tiles, updated=())
    def list_tiles(
        self,
        *,
        filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[TileRecord]:
        tiles = list(self._tiles.values())[:limit]
        result = []
        for t in tiles:
            row = {**t, "html_content": ""}
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
        self._dashboard_counter += 1
        token = f"sim_dash_{self._dashboard_counter:04d}"
        tile_list = tiles or []
        now = datetime.now(timezone.utc).isoformat()

        self._dashboards[token] = {
            "dashboard_id": self._dashboard_counter,
            "token": token,
            "title": title,
            "description": description,
            "layout": serialize_layout(tile_list),
            "tile_count": len(tile_list),
            "created_at": now,
            "updated_at": now,
        }

        return DashboardResult(
            url=f"https://simulated-console.example.com/dashboard/view/{token}",
            token=token,
            title=title,
            tiles=tile_list,
        )

    @functools.wraps(BaseDashboardManager.get_dashboard, updated=())
    def get_dashboard(self, token: str) -> Optional[DashboardResult]:
        data = self._dashboards.get(token)
        if data is None:
            return None
        tile_positions = deserialize_layout(data.get("layout", "[]"))
        return DashboardResult(
            url=f"https://simulated-console.example.com/dashboard/view/{token}",
            token=token,
            title=data.get("title"),
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
        if token not in self._dashboards:
            return DashboardResult(error=f"Dashboard '{token}' not found")

        data = self._dashboards[token]
        if title is not None:
            data["title"] = title
        if description is not None:
            data["description"] = description
        if tiles is not None:
            data["layout"] = serialize_layout(tiles)
            data["tile_count"] = len(tiles)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()

        final_tiles = tiles if tiles is not None else deserialize_layout(data["layout"])

        return DashboardResult(
            url=f"https://simulated-console.example.com/dashboard/view/{token}",
            token=token,
            title=data["title"],
            tiles=final_tiles,
        )

    @functools.wraps(BaseDashboardManager.delete_dashboard, updated=())
    def delete_dashboard(self, token: str) -> bool:
        return self._dashboards.pop(token, None) is not None

    @functools.wraps(BaseDashboardManager.list_dashboards, updated=())
    def list_dashboards(
        self,
        *,
        filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[DashboardRecord]:
        dashboards = list(self._dashboards.values())[:limit]
        return [DashboardRecord(**d) for d in dashboards]
