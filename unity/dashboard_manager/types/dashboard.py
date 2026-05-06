"""Dashboard layout type definitions for DashboardManager.

Defines Pydantic models for dashboard records (stored in Unify contexts)
and dashboard results (returned to the actor).
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from unity.common.authorship import AuthoredRow


class TilePosition(BaseModel):
    """Position and size of a tile within a dashboard grid (12-column layout).

    Attributes
    ----------
    tile_token : str
        Token of the tile to place at this position.
    x : int
        Column offset (0–11).
    y : int
        Row offset (units depend on grid row height).
    w : int
        Width in columns (1–12).
    h : int
        Height in row units.
    """

    tile_token: str
    x: int = 0
    y: int = 0
    w: int = 6
    h: int = 4


class DashboardRecordRow(AuthoredRow):
    """Fields inserted into the Dashboards/Layouts Unify context.

    ``dashboard_id`` is omitted because it is auto-counted by the backend.
    """

    token: str = Field(description="Unique 12-char URL-safe token")
    title: str = Field(description="Human-readable dashboard title")
    description: Optional[str] = Field(
        default=None,
        description="Optional longer description of the dashboard",
    )
    layout: str = Field(
        description="JSON-serialized list of TilePosition objects",
        json_schema_extra={"unify_type": "str"},
    )
    tile_count: int = Field(
        default=0,
        description="Number of tiles in the dashboard",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 creation timestamp",
    )
    updated_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 last-update timestamp",
    )


class DashboardRecord(DashboardRecordRow):
    """Full dashboard record including the server-assigned dashboard_id."""

    dashboard_id: Optional[int] = Field(
        default=None,
        description="Auto-incremented dashboard identifier",
    )


class DashboardResult(BaseModel):
    """Result returned to the actor after creating or updating a dashboard.

    Attributes
    ----------
    url : str | None
        Shareable URL to view the dashboard.
    token : str | None
        The 12-char token identifying this dashboard.
    title : str | None
        Title of the dashboard.
    tiles : list[TilePosition] | None
        Tile layout positions (present on success).
    error : str | None
        Error message if the operation failed.
    """

    model_config = ConfigDict(populate_by_name=True)

    url: Optional[str] = None
    token: Optional[str] = None
    title: Optional[str] = None
    tiles: Optional[List[TilePosition]] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """True if the dashboard was created/updated successfully."""
        return self.url is not None and self.error is None
