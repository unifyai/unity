"""Tile-related type definitions for DashboardManager.

Defines Pydantic models for tile records (stored in Unify contexts)
and tile results (returned to the actor).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DataBinding(BaseModel):
    """A declared data source for a tile that uses live data via the bridge.

    When ``create_tile`` is called with data bindings, query param fields
    are automatically validated by dry-running them through
    ``DataManager.filter(limit=5)``.  If the context does not exist, a
    column is misspelled, or the filter expression is invalid, the tile
    is **not** stored and ``TileResult.error`` reports the problem.

    Attributes
    ----------
    context : str
        Unify context path the tile will query at render time.
    alias : str | None
        Optional short name the tile's JS code uses to reference this binding.
    filter : str | None
        Row filter expression, validated against the context at creation
        time.  Same syntax as ``primitives.data.filter(filter=...)``.
    columns : list[str] | None
        Column names to return.  Validated against the context at creation.
    exclude_columns : list[str] | None
        Column names to omit.  Validated against the context at creation.
    order_by : str | None
        Column to sort by.  Validated against the context at creation.
    descending : bool
        Sort direction when ``order_by`` is set.  Default ``False``.
    """

    context: str
    alias: Optional[str] = None
    filter: Optional[str] = None
    columns: Optional[List[str]] = None
    exclude_columns: Optional[List[str]] = None
    order_by: Optional[str] = None
    descending: bool = False


class TileRecordRow(BaseModel):
    """Fields inserted into the Dashboards/Tiles Unify context.

    ``tile_id`` is omitted because it is auto-counted by the backend.
    """

    token: str = Field(description="Unique 12-char URL-safe token")
    title: str = Field(description="Human-readable tile title")
    description: Optional[str] = Field(
        default=None,
        description="Optional longer description of the tile",
    )
    html_content: str = Field(
        description="Self-contained HTML visualization content",
        json_schema_extra={"unify_type": "str"},
    )
    has_data_bindings: bool = Field(
        default=False,
        description="Whether the tile uses UnifyData.query() for live data",
    )
    data_binding_contexts: Optional[str] = Field(
        default=None,
        description="Comma-separated Unify context paths for data bindings",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 creation timestamp",
    )
    updated_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 last-update timestamp",
    )


class TileRecord(TileRecordRow):
    """Full tile record including the server-assigned tile_id."""

    tile_id: Optional[int] = Field(
        default=None,
        description="Auto-incremented tile identifier",
    )


class TileResult(BaseModel):
    """Result returned to the actor after creating or updating a tile.

    Attributes
    ----------
    url : str | None
        Shareable URL to view the tile (e.g., ``/tile/view/{token}``).
    token : str | None
        The 12-char token identifying this tile.
    title : str | None
        Title of the tile.
    error : str | None
        Error message if the operation failed.
    """

    model_config = ConfigDict(populate_by_name=True)

    url: Optional[str] = None
    token: Optional[str] = None
    title: Optional[str] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """True if the tile was created/updated successfully."""
        return self.url is not None and self.error is None
