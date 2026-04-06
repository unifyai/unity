"""Dashboard manager type definitions."""

from unity.dashboard_manager.types.tile import (
    DataBinding,
    TileRecord,
    TileRecordRow,
    TileResult,
)
from unity.dashboard_manager.types.dashboard import (
    DashboardRecord,
    DashboardRecordRow,
    DashboardResult,
    TilePosition,
)

__all__ = [
    "DataBinding",
    "TileRecord",
    "TileRecordRow",
    "TileResult",
    "DashboardRecord",
    "DashboardRecordRow",
    "DashboardResult",
    "TilePosition",
]
