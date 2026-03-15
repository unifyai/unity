"""
Table view type definitions for DataManager.

This module defines Pydantic models for table view configuration and results.
These types are used by DataManager's table_view() method to create shareable
table excerpts via the Table View API.
"""

from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, ConfigDict, Field


class TableViewConfig(BaseModel):
    """
    Configuration for a table view.

    Maps to the table_config in the Table View API request body.

    Attributes
    ----------
    columns_visible : list[str] | None
        Columns to show. If None, all columns are visible.
    columns_hidden : list[str] | None
        Columns to hide. Alternative to columns_visible.
    columns_order : list[str] | None
        Custom column ordering.
    row_limit : int | None
        Maximum number of rows to display.
    sort_by : str | None
        Column to sort by.
    sort_order : str | None
        Sort direction: "asc" or "desc".
    """

    columns_visible: Optional[List[str]] = None
    columns_hidden: Optional[List[str]] = None
    columns_order: Optional[List[str]] = None
    row_limit: Optional[int] = None
    sort_by: Optional[str] = None
    sort_order: Optional[str] = None


class TableViewResult(BaseModel):
    """
    Result of a table view creation attempt.

    Contains either a successful URL or error information if creation failed.

    Attributes
    ----------
    url : str | None
        URL to the generated table view. None if creation failed.
    token : str | None
        Token for accessing the table view.
    title : str | None
        Title of the table view.
    context : str | None
        Source context path for the table data.
    error : str | None
        Error message if creation failed.
    traceback_str : str | None
        Full traceback string for debugging failures.

    Properties
    ----------
    succeeded : bool
        True if the table view was created successfully.
    """

    model_config = ConfigDict(populate_by_name=True)

    url: Optional[str] = None
    token: Optional[str] = None
    title: Optional[str] = None
    context: Optional[str] = None
    error: Optional[str] = None
    traceback_str: Optional[str] = Field(default=None, alias="traceback")

    @property
    def succeeded(self) -> bool:
        """True if the table view was created successfully."""
        return self.url is not None and self.error is None
