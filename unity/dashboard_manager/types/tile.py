"""Tile-related type definitions for DashboardManager.

Defines Pydantic models for tile records (stored in Unify contexts),
tile results (returned to the actor), and data binding types that
declare what live queries a tile makes at render time.
"""

from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Data binding types (discriminated union via ``operation``)
# ---------------------------------------------------------------------------


class FilterBinding(BaseModel):
    """Single-context row fetch, validated via ``DataManager.filter(limit=5)``.

    Declares a live data source that fetches rows from a single Unify context
    with optional filtering, column selection, sorting, and pagination.
    """

    operation: Literal["filter"] = "filter"
    context: str
    alias: Optional[str] = None
    filter: Optional[str] = None
    columns: Optional[List[str]] = None
    exclude_columns: Optional[List[str]] = None
    order_by: Optional[str] = None
    descending: bool = False
    limit: Optional[int] = None
    offset: Optional[int] = None
    group_by: Optional[List[str]] = None


class ReduceBinding(BaseModel):
    """Single-context aggregation, validated via ``DataManager.reduce()``.

    Declares a live data source that computes an aggregate metric (count, sum,
    avg, min, max, etc.) over a single Unify context, optionally grouped.
    """

    operation: Literal["reduce"] = "reduce"
    context: str
    alias: Optional[str] = None
    metric: str
    columns: Union[str, List[str]]
    filter: Optional[str] = None
    group_by: Optional[Union[str, List[str]]] = None
    result_where: Optional[str] = None


class JoinBinding(BaseModel):
    """Cross-context join, validated via ``DataManager.filter_join(result_limit=5)``.

    Declares a live data source that joins two Unify contexts and returns
    the resulting rows with optional post-join filtering and pagination.
    """

    operation: Literal["join"] = "join"
    tables: List[str]
    alias: Optional[str] = None
    join_expr: str
    select: Dict[str, str]
    mode: str = "inner"
    left_where: Optional[str] = None
    right_where: Optional[str] = None
    result_where: Optional[str] = None
    result_limit: int = 100
    result_offset: int = 0


class JoinReduceBinding(BaseModel):
    """Cross-context join + aggregation, validated via ``DataManager.reduce_join()``.

    Declares a live data source that joins two Unify contexts and computes
    an aggregate metric over the joined result, optionally grouped.
    """

    operation: Literal["join_reduce"] = "join_reduce"
    tables: List[str]
    alias: Optional[str] = None
    join_expr: str
    select: Dict[str, str]
    mode: str = "inner"
    left_where: Optional[str] = None
    right_where: Optional[str] = None
    metric: str
    columns: Union[str, List[str]]
    group_by: Optional[Union[str, List[str]]] = None
    result_where: Optional[str] = None


DataBinding = Annotated[
    Union[FilterBinding, ReduceBinding, JoinBinding, JoinReduceBinding],
    Field(discriminator="operation"),
]


# ---------------------------------------------------------------------------
# Tile storage and result types
# ---------------------------------------------------------------------------


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
        description="Whether the tile uses the live data bridge",
    )
    data_binding_contexts: Optional[str] = Field(
        default=None,
        description="Comma-separated Unify context paths for data bindings",
    )
    on_data_script: Optional[str] = Field(
        default=None,
        description=(
            "JS code block executed with a `data` variable in scope. "
            "Console wraps this as (function(data){ ... })(results) after "
            "auto-executing the serialized data bindings."
        ),
    )
    data_bindings_json: Optional[str] = Field(
        default=None,
        description=(
            "JSON-serialized list of binding dicts (each with an `operation` "
            "discriminator). Console reads this to auto-generate bridge "
            "calls at render time."
        ),
    )
    created_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 creation timestamp",
    )
    updated_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 last-update timestamp",
    )
    authoring_assistant_id: Optional[int] = Field(
        default=None,
        description=(
            "Assistant id of the body that authored this tile. Stamped once "
            "at create time and preserved across updates so reviewers can "
            "attribute Hive-shared rows to their authoring body."
        ),
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
