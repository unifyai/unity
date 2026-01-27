"""
Utility functions for embedding-based vector search through the logs.
"""

import hashlib
import os
import sys
import tempfile
import unify
import threading
from contextlib import contextmanager

# Cross-platform file locking
if sys.platform == "win32":
    import msvcrt

    def _lock_file(file_obj):
        """Acquire an exclusive lock on the file (Windows)."""
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock_file(file_obj):
        """Release the lock on the file (Windows)."""
        file_obj.seek(0)
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_file(file_obj):
        """Acquire an exclusive lock on the file (Unix)."""
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_file(file_obj):
        """Release the lock on the file (Unix)."""
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

# Model to use for text embeddings
EMBED_MODEL = "text-embedding-3-small"


# In‑process locks keyed by (context, column_key) to avoid race conditions when
# multiple concurrent tool calls attempt to create the same derived/embedding
# columns at the same time.
_COLUMN_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_COLUMN_LOCKS_LOCK = threading.Lock()


def _get_column_lock(context: str, key: str) -> threading.Lock:
    """Return a process-local lock for a specific (context, key)."""
    lk_key = (context, key)
    with _COLUMN_LOCKS_LOCK:
        lock = _COLUMN_LOCKS.get(lk_key)
        if lock is None:
            lock = threading.Lock()
            _COLUMN_LOCKS[lk_key] = lock
        return lock


@contextmanager
def _cross_process_column_lock(context: str, key: str, timeout: float = 600.0):
    """
    Cross-process file lock for column creation.

    Provides coordination across parallel test processes that share the same
    Unify backend. Uses a file lock in the temp directory keyed by a hash of
    (context, key) to avoid filename length issues.

    The overhead is ~12μs per lock acquisition - negligible compared to the
    ~500ms+ embedding operations being protected.

    Parameters
    ----------
    context : str
        The Unify context name.
    key : str
        The column key being created.
    timeout : float, default 600.0
        Maximum seconds to wait for the lock (10 minutes). If exceeded, raises TimeoutError.
        This prevents indefinite hangs if another process holds the lock and is
        stuck (e.g., hung API call).
    """
    import time

    # Hash the key to avoid filesystem issues with long/special characters
    lock_id = hashlib.sha1(f"{context}:{key}".encode()).hexdigest()[:16]
    lock_path = os.path.join(tempfile.gettempdir(), f"unity_col_{lock_id}.lock")

    lock_file = open(lock_path, "w")
    try:
        # Try non-blocking acquisition with timeout to avoid indefinite hangs
        start = time.monotonic()
        while True:
            try:
                _lock_file(lock_file)
                break  # Successfully acquired lock
            except (BlockingIOError, OSError):
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    lock_file.close()
                    raise TimeoutError(
                        f"Timeout after {timeout}s waiting for column lock: "
                        f"context={context}, key={key}. Another process may be hung.",
                    )
                time.sleep(0.1)  # Brief sleep before retry
        yield
    finally:
        _unlock_file(lock_file)
        lock_file.close()


def list_private_fields(context: str) -> list[str]:
    """
    Return a list of private field names for a context.

    Private fields are defined as columns whose names start with "_". These
    typically include derived/debug columns and embedding vectors which can be
    very large, so they should be excluded from payloads returned to clients.
    """
    try:
        fields = unify.get_fields(context=context)
        return [name for name in fields.keys() if name.startswith("_")]
    except Exception:
        # If field introspection fails (e.g. offline tests), fall back to none
        return []


def escape_single_quotes(text: str) -> str:
    """Return text with single quotes escaped for Unify expressions."""
    return text.replace("'", "\\'")


def ensure_derived_column(
    context: str,
    key: str,
    equation: str,
    *,
    referenced_logs_context: str | None = None,
    derived: bool | None = None,
    from_ids: list[int] | None = None,
) -> None:
    """
    Ensure a derived column exists with the given equation.

    - Creates the column if missing, guarded by a process-local lock to avoid
      duplicate creations under concurrency.
    - Tolerates backend uniqueness races.
    - By default, scopes placeholders to a local alias `lg` referencing the
      provided `context` when `referenced_logs_context` is not specified.
    """
    # Fast path: if field already exists and we are not doing targeted
    # derived logs creation using from_ids, return without locking or logging
    try:
        fields = unify.get_fields(context=context)
        if key in fields and not from_ids:
            return
    except Exception:
        # If introspection fails, fall through to locked creation
        pass

    # Attempt creation under both a cross-process file lock (for parallel test
    # isolation) and a process-local thread lock (for in-process concurrency).
    # The file lock coordinates across parallel pytest sessions; the thread lock
    # handles concurrent async tasks within a single process.
    thread_lock = _get_column_lock(context, key)
    with _cross_process_column_lock(context, key):
        with thread_lock:
            # Early return if the column already exists and
            # we are not doing targeted derived logs creation using from_ids
            # We intentionally do this to avoid redundant calls to create
            # duplicate embeddings that will get rejected by the backend
            # due to duplication constraint
            existing = unify.get_fields(context=context)
            if key in existing and not from_ids:
                return

            try:
                try:
                    fields = unify.get_fields(context=context)
                    if key in fields and not from_ids:
                        return
                except Exception:
                    pass

                referenced_logs = {}
                if from_ids:
                    # Instruct backend to scope the operation to a subset of log entries
                    referenced_logs = {
                        "lg": list(from_ids),
                    }
                else:
                    referenced_logs = {
                        "lg": {"context": referenced_logs_context or context},
                    }

                response = unify.create_derived_logs(
                    context=context,
                    key=key,
                    equation=equation,
                    referenced_logs=referenced_logs,
                    derived=derived,
                )
                # Be quiet in normal operation; tests assert no failure logs appear.
                # print(f"{response}")
            except unify.RequestError as e:
                body = getattr(e.response, "text", "") or ""
                # Treat duplicate/exists as success and do not emit error output
                if (
                    "already exists" in body
                    or "duplicate key value violates unique constraint" in body
                ):
                    return
                # For other errors, re-raise
                raise e


def ensure_vector_column(
    context: str,
    embed_column: str,
    source_column: str,
    derived_expr: str | None = None,
    *,
    from_ids: list[int] | None = None,
) -> None:
    """
    Ensure that a vector column exists in the given context. If it does not,
    create a derived column using the embed() function with the defined embedding model.

    Args:
        context (str): The Unify context (e.g., "Knowledge/table_name" or "ContextName").
        embed_column (str): The name of the vector column to ensure. (eg: "content_emb")
        source_column (str): The name of the source column to embed. (eg: "content_plus_desc")
        derived_expr Optional(str): An optional expression to dynamically derive the source column
            (in case it's not already present) (eg: "str({name}) + ' || ' + str({description})")
    """
    # If a derived expression was provided for the source, ensure the source column exists.
    if derived_expr is not None:
        # Scope placeholder references to the local logs alias
        derived_expr = derived_expr.replace("{", "{lg:")
        ensure_derived_column(
            context=context,
            key=source_column,
            equation=derived_expr,
            from_ids=from_ids,
        )

    # Early return if the embedding column already exists and
    # we are not doing targetted derived logs creation using from_ids
    existing = unify.get_fields(context=context)
    if embed_column in existing and not from_ids:
        return

    # Define the embedding equation with explicit lg scoping and ensure the embedding column.
    embed_expr = f"embed({{lg:{source_column}}}, model='{EMBED_MODEL}')"
    ensure_derived_column(
        context=context,
        key=embed_column,
        equation=embed_expr,
        from_ids=from_ids,
    )
    return None
