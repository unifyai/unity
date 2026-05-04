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

from unity.common.context_registry import (
    PERSONAL_DESTINATION,
    SPACE_CONTEXT_PREFIX,
    SPACE_DESTINATION_PREFIX,
)
from unity.common.join_utils import rewrite_join_paths
from unity.dashboard_manager.base import DASHBOARD_DATA_SCOPE, BaseDashboardManager
from unity.dashboard_manager.ops.dashboard_ops import (
    serialize_layout,
    deserialize_layout,
)
from unity.dashboard_manager.types.dashboard import (
    DashboardRecord,
    DashboardResult,
    TilePosition,
)
from unity.dashboard_manager.ops.tile_ops import (
    _contexts_for_binding,
    ensure_binding_aliases,
    resolve_binding_contexts,
    serialize_bindings,
    validate_on_data,
)
from unity.dashboard_manager.types.tile import (
    DataBinding,
    FilterBinding,
    JoinBinding,
    JoinReduceBinding,
    ReduceBinding,
    TileRecord,
    TileResult,
)
from unity.session_details import SESSION_DETAILS


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

    def _validate_data_scope(self, data_scope: str) -> None:
        """Validate the tile data-source scope used by simulated primitives."""
        if data_scope == DASHBOARD_DATA_SCOPE:
            return
        self._normalize_destination(data_scope)

    def _normalize_destination(self, destination: str | None) -> str:
        """Return the canonical row destination for simulated storage."""
        if destination in (None, PERSONAL_DESTINATION):
            return PERSONAL_DESTINATION
        if (
            destination.startswith(SPACE_DESTINATION_PREFIX)
            and destination[len(SPACE_DESTINATION_PREFIX) :].isdigit()
        ):
            space_id = int(destination[len(SPACE_DESTINATION_PREFIX) :])
            if space_id in SESSION_DETAILS.space_ids:
                return destination
        raise ValueError(
            "invalid_destination: destination must be personal or space:<id>.",
        )

    def _visible_destinations(self) -> set[str]:
        """Return the destinations readable by the simulated assistant."""
        return {
            PERSONAL_DESTINATION,
            *[
                f"{SPACE_DESTINATION_PREFIX}{space_id}"
                for space_id in SESSION_DETAILS.space_ids
            ],
        }

    def _binding_base_context(
        self,
        *,
        row_destination: str,
        data_scope: str,
    ) -> str | None:
        """Return the shared root that fresh simulated bindings should use."""
        binding_destination = (
            row_destination if data_scope == DASHBOARD_DATA_SCOPE else data_scope
        )
        if binding_destination.startswith(SPACE_DESTINATION_PREFIX):
            space_id = binding_destination[len(SPACE_DESTINATION_PREFIX) :]
            return f"Spaces/{space_id}"
        return None

    def _resolve_binding_contexts(
        self,
        bindings: List[DataBinding],
        *,
        row_destination: str,
        data_scope: str,
    ) -> List[DataBinding]:
        """Resolve simulated bindings against the same shared root policy."""
        base_context = self._binding_base_context(
            row_destination=row_destination,
            data_scope=data_scope,
        )
        try:
            resolved = resolve_binding_contexts(bindings, base_context=base_context)
        except Exception:
            resolved = bindings
        if not base_context:
            return resolved

        def scoped(path: str) -> str:
            path = path.strip().lstrip("/")
            if path.startswith(SPACE_CONTEXT_PREFIX):
                return path
            return f"{base_context}/{path}"

        scoped_bindings: List[DataBinding] = []
        for binding in resolved:
            if isinstance(binding, (FilterBinding, ReduceBinding)):
                binding = binding.model_copy(
                    update={"context": scoped(binding.context)},
                )
            elif isinstance(binding, (JoinBinding, JoinReduceBinding)):
                scoped_tables = [scoped(table) for table in binding.tables]
                join_expr, select = rewrite_join_paths(
                    list(binding.tables),
                    scoped_tables,
                    binding.join_expr,
                    dict(binding.select),
                )
                binding = binding.model_copy(
                    update={
                        "tables": scoped_tables,
                        "join_expr": join_expr,
                        "select": select,
                    },
                )
            scoped_bindings.append(binding)
        return scoped_bindings

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
        on_data: Optional[str] = None,
        destination: str | None = None,
        data_scope: str = DASHBOARD_DATA_SCOPE,
    ) -> TileResult:
        try:
            row_destination = self._normalize_destination(destination)
            validate_on_data(on_data, data_bindings)
            if data_scope != DASHBOARD_DATA_SCOPE and not data_bindings:
                raise ValueError(
                    "data_scope can only be set when fresh data_bindings are supplied.",
                )
            self._validate_data_scope(data_scope)
        except Exception as exc:
            return TileResult(error=str(exc))

        self._tile_counter += 1
        token = f"sim_tile_{self._tile_counter:04d}"
        has_bindings = bool(data_bindings)
        binding_contexts = None
        bindings_json = None

        if data_bindings:
            if on_data is not None:
                data_bindings = ensure_binding_aliases(data_bindings)
            try:
                data_bindings = self._resolve_binding_contexts(
                    data_bindings,
                    row_destination=row_destination,
                    data_scope=data_scope,
                )
            except Exception:
                pass
            all_ctxs: list[str] = []
            for b in data_bindings:
                all_ctxs.extend(_contexts_for_binding(b))
            binding_contexts = ",".join(dict.fromkeys(all_ctxs))
            bindings_json = serialize_bindings(data_bindings)

        now = datetime.now(timezone.utc).isoformat()

        self._tiles[token] = {
            "tile_id": self._tile_counter,
            "token": token,
            "title": title,
            "description": description,
            "html_content": html,
            "destination": row_destination,
            "has_data_bindings": has_bindings,
            "data_scope": data_scope,
            "data_binding_contexts": binding_contexts,
            "on_data_script": on_data,
            "data_bindings_json": bindings_json,
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
        if (
            data is None
            or data.get("destination", "personal") not in self._visible_destinations()
        ):
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
        data_bindings: Optional[List[DataBinding]] = None,
        on_data: Optional[str] = None,
        destination: str | None = None,
        data_scope: Optional[str] = None,
    ) -> TileResult:
        try:
            row_destination = self._normalize_destination(destination)
        except Exception as exc:
            return TileResult(error=str(exc))
        if (
            token not in self._tiles
            or self._tiles[token].get("destination", "personal") != row_destination
        ):
            return TileResult(error=f"Tile '{token}' not found")
        if data_scope is not None and not data_bindings:
            return TileResult(
                error=(
                    "data_scope can only be changed when fresh data_bindings "
                    "are supplied."
                ),
            )
        if data_scope is not None:
            try:
                self._validate_data_scope(data_scope)
            except Exception as exc:
                return TileResult(error=str(exc))

        tile_data = self._tiles[token]
        if html is not None:
            tile_data["html_content"] = html
        if title is not None:
            tile_data["title"] = title
        if description is not None:
            tile_data["description"] = description

        if data_bindings is not None:
            if data_bindings:
                effective_data_scope = data_scope or tile_data.get(
                    "data_scope",
                    DASHBOARD_DATA_SCOPE,
                )
                if on_data is not None and on_data != "":
                    data_bindings = ensure_binding_aliases(data_bindings)
                try:
                    data_bindings = self._resolve_binding_contexts(
                        data_bindings,
                        row_destination=row_destination,
                        data_scope=effective_data_scope,
                    )
                except Exception:
                    pass
                all_ctxs: list[str] = []
                for b in data_bindings:
                    all_ctxs.extend(_contexts_for_binding(b))
                tile_data["has_data_bindings"] = True
                tile_data["data_scope"] = effective_data_scope
                tile_data["data_binding_contexts"] = ",".join(
                    dict.fromkeys(all_ctxs),
                )
                tile_data["data_bindings_json"] = serialize_bindings(
                    data_bindings,
                )
            else:
                tile_data["has_data_bindings"] = False
                tile_data["data_scope"] = DASHBOARD_DATA_SCOPE
                tile_data["data_binding_contexts"] = None
                tile_data["data_bindings_json"] = None

        if on_data is not None:
            if on_data == "":
                tile_data["on_data_script"] = None
            else:
                has_bindings = bool(data_bindings) or tile_data.get(
                    "has_data_bindings",
                    False,
                )
                if not has_bindings:
                    validate_on_data(on_data, data_bindings)
                tile_data["on_data_script"] = on_data

        tile_data["updated_at"] = datetime.now(timezone.utc).isoformat()

        return TileResult(
            url=f"https://simulated-console.example.com/tile/view/{token}",
            token=token,
            title=tile_data["title"],
        )

    @functools.wraps(BaseDashboardManager.delete_tile, updated=())
    def delete_tile(self, token: str, *, destination: str | None = None) -> bool:
        try:
            row_destination = self._normalize_destination(destination)
        except ValueError:
            return False
        data = self._tiles.get(token)
        if data is None or data.get("destination", "personal") != row_destination:
            return False
        self._tiles.pop(token)
        return True

    @functools.wraps(BaseDashboardManager.list_tiles, updated=())
    def list_tiles(
        self,
        *,
        filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[TileRecord]:
        visible = self._visible_destinations()
        tiles = [
            tile
            for tile in self._tiles.values()
            if tile.get("destination", "personal") in visible
        ][:limit]
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
        destination: str | None = None,
    ) -> DashboardResult:
        try:
            row_destination = self._normalize_destination(destination)
        except Exception as exc:
            return DashboardResult(error=str(exc))
        self._dashboard_counter += 1
        token = f"sim_dash_{self._dashboard_counter:04d}"
        tile_list = tiles or []
        now = datetime.now(timezone.utc).isoformat()

        self._dashboards[token] = {
            "dashboard_id": self._dashboard_counter,
            "token": token,
            "title": title,
            "description": description,
            "destination": row_destination,
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
        if (
            data is None
            or data.get("destination", "personal") not in self._visible_destinations()
        ):
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
        destination: str | None = None,
    ) -> DashboardResult:
        try:
            row_destination = self._normalize_destination(destination)
        except Exception as exc:
            return DashboardResult(error=str(exc))
        if (
            token not in self._dashboards
            or self._dashboards[token].get("destination", "personal") != row_destination
        ):
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
    def delete_dashboard(self, token: str, *, destination: str | None = None) -> bool:
        try:
            row_destination = self._normalize_destination(destination)
        except ValueError:
            return False
        data = self._dashboards.get(token)
        if data is None or data.get("destination", "personal") != row_destination:
            return False
        self._dashboards.pop(token)
        return True

    @functools.wraps(BaseDashboardManager.list_dashboards, updated=())
    def list_dashboards(
        self,
        *,
        filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[DashboardRecord]:
        visible = self._visible_destinations()
        dashboards = [
            dashboard
            for dashboard in self._dashboards.values()
            if dashboard.get("destination", "personal") in visible
        ][:limit]
        return [DashboardRecord(**d) for d in dashboards]
