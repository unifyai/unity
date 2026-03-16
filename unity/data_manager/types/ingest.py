"""Ingest-related type definitions for DataManager.

This module defines Pydantic models for configuring and reporting the
results of :meth:`DataManager.ingest` operations.

Terminology
-----------
* **Ingest (narrow):** Create a table context and insert rows. No parsing,
  no embedding. This is the core behaviour of ``DataManager.ingest()``.
* **Embed:** Create vector embeddings for text columns. A separate, optional
  step within ``ingest()`` that only runs when ``embed_columns`` is provided.

See Also
--------
DataManager.ingest : The public method these types support.
DataManager.create_table : Low-level table creation.
DataManager.insert_rows : Low-level row insertion.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class IngestExecutionConfig(BaseModel):
    """Advanced execution options for :meth:`DataManager.ingest`.

    These settings control the internal pipeline engine that ``ingest()``
    uses for chunked parallel row insertion and embedding.  Default values
    are suitable for most workloads; tweak them for very large datasets or
    rate-limited backends.

    Attributes
    ----------
    max_workers : int
        Maximum threads used for concurrent chunk processing (insertion and
        embedding).  Higher values increase throughput at the cost of more
        backend pressure.  Default ``4``.
    max_retries : int
        Number of retry attempts per failed chunk before giving up.
        Uses exponential backoff (see *retry_delay_seconds*).  Default ``3``.
    retry_delay_seconds : float
        Base delay between retries in seconds.  Actual delay is
        ``retry_delay_seconds * 2 ** (attempt - 1)``.  Default ``3.0``.
    fail_fast : bool
        If ``True``, stop the pipeline on the first chunk failure rather
        than attempting remaining chunks.  Default ``False``.

    Examples
    --------
    >>> from unity.data_manager.types import IngestExecutionConfig
    >>> cfg = IngestExecutionConfig(max_workers=8, fail_fast=True)
    >>> dm.ingest("Data/Sales", rows, execution=cfg)
    """

    max_workers: int = Field(default=4, ge=1)
    max_retries: int = Field(default=3, ge=0)
    retry_delay_seconds: float = Field(default=3.0, ge=0.0)
    fail_fast: bool = False


class IngestResult(BaseModel):
    """Result of a :meth:`DataManager.ingest` call.

    Captures what the pipeline accomplished: how many rows were inserted and
    embedded, how many chunks were processed, and the total wall-clock time.

    Attributes
    ----------
    context : str
        The Unify context path that was written to.
    rows_inserted : int
        Total number of rows successfully inserted across all chunks.
    rows_embedded : int
        Total number of rows for which embeddings were created.  ``0`` when
        ``embed_columns`` was not provided.
    log_ids : list[int]
        Backend log IDs for each ``insert_rows`` call (useful for auditing).
    duration_ms : float
        Total wall-clock time for the entire ingest operation in milliseconds.
    chunks_processed : int
        Number of row chunks that were inserted (each chunk is a separate
        backend call).

    Examples
    --------
    >>> result = dm.ingest("Data/MH/Repairs", rows)
    >>> print(f"Inserted {result.rows_inserted} rows in {result.duration_ms:.0f}ms")
    """

    context: str
    rows_inserted: int = 0
    rows_embedded: int = 0
    log_ids: List[int] = Field(default_factory=list)
    duration_ms: float = 0.0
    chunks_processed: int = 0
