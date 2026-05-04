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

from unity.common.context_registry import (
    ContextRegistry,
    INVALID_DESTINATION_ERROR,
    SPACE_DESTINATION_PREFIX,
    TableContext,
)
from unity.common.model_to_fields import model_to_fields
from unity.common.tool_outcome import ToolErrorException
from unity.dashboard_manager.base import DASHBOARD_DATA_SCOPE, BaseDashboardManager
from unity.dashboard_manager.ops.dashboard_ops import (
    build_dashboard_record_row,
    deserialize_layout,
    serialize_layout,
)
from unity.dashboard_manager.ops.tile_ops import (
    _contexts_for_binding,
    build_tile_record_row,
    ensure_binding_aliases,
    resolve_binding_contexts,
    serialize_bindings,
    validate_data_bindings,
    validate_on_data,
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
from unity.session_details import SESSION_DETAILS
from unity.settings import SETTINGS

logger = logging.getLogger(__name__)

TILES_TABLE = "Dashboards/Tiles"
LAYOUTS_TABLE = "Dashboards/Layouts"


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
        logger.debug("DashboardManager initialized")

    def _get_dm(self):
        from unity.manager_registry import ManagerRegistry

        return ManagerRegistry.get_data_manager()

    def _table_context_for_root(self, root_context: str, table_name: str) -> str:
        """Return the concrete dashboard context under one registry root."""
        return f"{root_context.strip('/')}/{table_name}"

    def _table_context_for_destination(
        self,
        table_name: str,
        destination: str | None,
    ) -> str:
        """Resolve a dashboard row destination into a concrete table context."""
        root_context = ContextRegistry.write_root(
            self,
            table_name,
            destination=destination,
        )
        return self._table_context_for_root(root_context, table_name)

    def _read_table_contexts(self, table_name: str) -> list[str]:
        """Return ordered concrete dashboard table contexts visible to this assistant."""
        return [
            self._table_context_for_root(root_context, table_name)
            for root_context in ContextRegistry.read_roots(self, table_name)
        ]

    def _resolved_root_context(
        self,
        table_name: str,
        destination: str | None,
    ) -> str:
        """Return the root context for a destination without provisioning."""
        _, _, root_context = ContextRegistry.resolve_root(
            self,
            table_name,
            destination=destination,
        )
        return root_context

    def _data_binding_root(
        self,
        table_name: str,
        *,
        destination: str | None,
        data_scope: str,
    ) -> str:
        """Resolve which root live tile data bindings should read from."""
        if data_scope == DASHBOARD_DATA_SCOPE:
            return self._resolved_root_context(table_name, destination)
        if not data_scope.startswith(SPACE_DESTINATION_PREFIX):
            raise ToolErrorException(
                {
                    "error_kind": INVALID_DESTINATION_ERROR,
                    "message": (
                        "data_scope must be 'dashboard' or an accessible "
                        "'space:<id>' destination."
                    ),
                    "details": {
                        "destination": data_scope,
                        "space_ids": SESSION_DETAILS.space_ids,
                        "table_name": table_name,
                    },
                },
            )
        return self._resolved_root_context(table_name, data_scope)

    def _tool_error_message(self, exc: ToolErrorException) -> str:
        """Convert structured tool errors into Dashboard result errors."""
        return f"{exc.payload['error_kind']}: {exc.payload['message']}"

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
            tile_context = self._table_context_for_destination(
                TILES_TABLE,
                destination,
            )
            dm = self._get_dm()
            token = generate_token()
            bindings = validate_data_bindings(data_bindings)

            validate_on_data(on_data, bindings)

            if data_scope != DASHBOARD_DATA_SCOPE and not bindings:
                return TileResult(
                    error=(
                        "data_scope can only be set when fresh "
                        "data_bindings are supplied."
                    ),
                )

            if bindings and on_data is not None:
                bindings = ensure_binding_aliases(bindings)

            if bindings:
                binding_root = self._data_binding_root(
                    TILES_TABLE,
                    destination=destination,
                    data_scope=data_scope,
                )
                bindings = resolve_binding_contexts(
                    bindings,
                    base_context=binding_root,
                )
                verify_data_bindings(bindings, dm)

            row = build_tile_record_row(
                token=token,
                html=html,
                title=title,
                description=description,
                data_bindings=bindings,
                on_data=on_data,
                data_scope=data_scope,
            )

            dm.insert_rows(tile_context, [row.model_dump()])

            register_token(token, "tile", tile_context, _get_active_project())

            return TileResult(
                url=_build_tile_url(token),
                token=token,
                title=title,
            )
        except ToolErrorException as e:
            return TileResult(error=self._tool_error_message(e))
        except Exception as e:
            logger.exception("create_tile failed")
            return TileResult(error=str(e))

    @functools.wraps(BaseDashboardManager.get_tile, updated=())
    def get_tile(self, token: str) -> Optional[TileRecord]:
        dm = self._get_dm()
        for context in self._read_table_contexts(TILES_TABLE):
            rows = dm.filter(
                context,
                filter=f"token == '{token}'",
                limit=1,
            )
            if rows:
                return TileRecord(**rows[0])
        return None

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
            if data_scope is not None and not data_bindings:
                return TileResult(
                    error=(
                        "data_scope can only be changed when fresh "
                        "data_bindings are supplied."
                    ),
                )
            tile_context = self._table_context_for_destination(
                TILES_TABLE,
                destination,
            )
            dm = self._get_dm()
            updates: Dict[str, Any] = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if html is not None:
                updates["html_content"] = html
            if title is not None:
                updates["title"] = title
            if description is not None:
                updates["description"] = description

            if data_bindings is not None:
                bindings = validate_data_bindings(data_bindings)
                if bindings:
                    effective_data_scope = data_scope
                    if effective_data_scope is None:
                        existing_rows = dm.filter(
                            tile_context,
                            filter=f"token == '{token}'",
                            limit=1,
                        )
                        existing_scope = (
                            existing_rows[0].get("data_scope")
                            if existing_rows
                            else None
                        )
                        effective_data_scope = existing_scope or DASHBOARD_DATA_SCOPE
                    binding_root = self._data_binding_root(
                        TILES_TABLE,
                        destination=destination,
                        data_scope=effective_data_scope,
                    )
                    if on_data is not None and on_data != "":
                        bindings = ensure_binding_aliases(bindings)
                    bindings = resolve_binding_contexts(
                        bindings,
                        base_context=binding_root,
                    )
                    verify_data_bindings(bindings, dm)
                    updates["has_data_bindings"] = True
                    updates["data_scope"] = effective_data_scope

                    all_ctxs: list[str] = []
                    for b in bindings:
                        all_ctxs.extend(_contexts_for_binding(b))
                    updates["data_binding_contexts"] = ",".join(
                        dict.fromkeys(all_ctxs),
                    )
                    updates["data_bindings_json"] = serialize_bindings(bindings)
                else:
                    updates["has_data_bindings"] = False
                    updates["data_scope"] = DASHBOARD_DATA_SCOPE
                    updates["data_binding_contexts"] = None
                    updates["data_bindings_json"] = None

            if on_data is not None:
                if on_data == "":
                    updates["on_data_script"] = None
                else:
                    has_bindings = bool(data_bindings)
                    if not has_bindings:
                        existing_rows = dm.filter(
                            tile_context,
                            filter=f"token == '{token}'",
                            limit=1,
                        )
                        has_bindings = bool(
                            existing_rows and existing_rows[0].get("has_data_bindings"),
                        )
                    if not has_bindings:
                        validate_on_data(on_data, data_bindings)
                    updates["on_data_script"] = on_data

            updated_count = dm.update_rows(
                tile_context,
                updates,
                filter=f"token == '{token}'",
            )
            if updated_count == 0:
                return TileResult(error=f"Tile '{token}' not found")

            return TileResult(
                url=_build_tile_url(token),
                token=token,
                title=title,
            )
        except ToolErrorException as e:
            return TileResult(error=self._tool_error_message(e))
        except Exception as e:
            logger.exception("update_tile failed")
            return TileResult(error=str(e))

    @functools.wraps(BaseDashboardManager.delete_tile, updated=())
    def delete_tile(self, token: str, *, destination: str | None = None) -> bool:
        try:
            context = self._table_context_for_destination(TILES_TABLE, destination)
            dm = self._get_dm()
            deleted = dm.delete_rows(
                context,
                filter=f"token == '{token}'",
            )
            if deleted:
                delete_token(token)
            return deleted > 0
        except ToolErrorException:
            return False

    @functools.wraps(BaseDashboardManager.list_tiles, updated=())
    def list_tiles(
        self,
        *,
        filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[TileRecord]:
        dm = self._get_dm()
        result = []
        for context in self._read_table_contexts(TILES_TABLE):
            remaining = limit - len(result)
            if remaining <= 0:
                break
            rows = dm.filter(
                context,
                filter=filter,
                exclude_columns=["html_content"],
                limit=remaining,
            )
            for row in rows:
                row.setdefault("html_content", "")
                result.append(TileRecord(**row))
                if len(result) >= limit:
                    return result
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
            layout_context = self._table_context_for_destination(
                LAYOUTS_TABLE,
                destination,
            )
            token = generate_token()
            tile_list = tiles or []
            row = build_dashboard_record_row(
                token=token,
                title=title,
                tiles=tile_list,
                description=description,
            )

            dm = self._get_dm()
            dm.insert_rows(layout_context, [row.model_dump()])

            register_token(token, "dashboard", layout_context, _get_active_project())

            return DashboardResult(
                url=_build_dashboard_url(token),
                token=token,
                title=title,
                tiles=tile_list,
            )
        except ToolErrorException as e:
            return DashboardResult(error=self._tool_error_message(e))
        except Exception as e:
            logger.exception("create_dashboard failed")
            return DashboardResult(error=str(e))

    @functools.wraps(BaseDashboardManager.get_dashboard, updated=())
    def get_dashboard(self, token: str) -> Optional[DashboardResult]:
        dm = self._get_dm()
        for context in self._read_table_contexts(LAYOUTS_TABLE):
            rows = dm.filter(
                context,
                filter=f"token == '{token}'",
                limit=1,
            )
            if rows:
                record = rows[0]
                tile_positions = deserialize_layout(record.get("layout", "[]"))
                return DashboardResult(
                    url=_build_dashboard_url(token),
                    token=token,
                    title=record.get("title"),
                    tiles=tile_positions,
                )
        return None

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
            layout_context = self._table_context_for_destination(
                LAYOUTS_TABLE,
                destination,
            )
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
            updated_count = dm.update_rows(
                layout_context,
                updates,
                filter=f"token == '{token}'",
            )
            if updated_count == 0:
                return DashboardResult(error=f"Dashboard '{token}' not found")

            final_tiles = tiles
            if final_tiles is None:
                rows = dm.filter(
                    layout_context,
                    filter=f"token == '{token}'",
                    limit=1,
                )
                final_tiles = (
                    deserialize_layout(rows[0].get("layout", "[]")) if rows else []
                )

            return DashboardResult(
                url=_build_dashboard_url(token),
                token=token,
                title=title,
                tiles=final_tiles,
            )
        except ToolErrorException as e:
            return DashboardResult(error=self._tool_error_message(e))
        except Exception as e:
            logger.exception("update_dashboard failed")
            return DashboardResult(error=str(e))

    @functools.wraps(BaseDashboardManager.delete_dashboard, updated=())
    def delete_dashboard(self, token: str, *, destination: str | None = None) -> bool:
        try:
            context = self._table_context_for_destination(LAYOUTS_TABLE, destination)
            dm = self._get_dm()
            deleted = dm.delete_rows(
                context,
                filter=f"token == '{token}'",
            )
            if deleted:
                delete_token(token)
            return deleted > 0
        except ToolErrorException:
            return False

    @functools.wraps(BaseDashboardManager.list_dashboards, updated=())
    def list_dashboards(
        self,
        *,
        filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[DashboardRecord]:
        dm = self._get_dm()
        result: list[DashboardRecord] = []
        for context in self._read_table_contexts(LAYOUTS_TABLE):
            remaining = limit - len(result)
            if remaining <= 0:
                break
            rows = dm.filter(
                context,
                filter=filter,
                limit=remaining,
            )
            for row in rows:
                result.append(DashboardRecord(**row))
                if len(result) >= limit:
                    return result
        return result
