"""
Utility functions for embedding-based vector search through the logs.
"""

import os

import requests
import unify
import threading

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


def ensure_vector_column(
    context: str,
    embed_column: str,
    source_column: str,
    derived_expr: str | None = None,
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
    # Retrieve existing columns and their types
    existing = unify.get_fields(context=context)
    if derived_expr is not None:
        # Scope placeholder references to the local logs alias
        derived_expr = derived_expr.replace("{", "{lg:")

    # Ensure the derived source column exists (when required)
    if source_column not in existing:
        if derived_expr is None:
            raise ValueError(
                f"Source column '{source_column}' does not exist in context '{context}' "
                f"and no derived_expr was provided to create it.",
            )
        source_lock = _get_column_lock(context, source_column)
        with source_lock:
            # Re-check after acquiring the lock in case another thread created it
            existing = unify.get_fields(context=context)
            if source_column not in existing:
                url = f"{os.environ['UNIFY_BASE_URL']}/logs/derived"
                headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
                expr = derived_expr
                json_input = {
                    "project": unify.active_project(),
                    "context": context,
                    "key": source_column,
                    "equation": expr,
                    "referenced_logs": {"lg": {"context": context}},
                }
                response = requests.request(
                    "POST",
                    url,
                    json=json_input,
                    headers=headers,
                )
                if response.status_code != 200:
                    # Tolerate backend uniqueness races from other processes
                    body = getattr(response, "text", "") or ""
                    if (
                        "already exists" in body
                        or "duplicate key value violates unique constraint" in body
                    ):
                        pass
                    else:
                        assert response.status_code == 200, response.text

    # Ensure the embedding column exists
    # Refresh existing fields view
    existing = unify.get_fields(context=context)
    if embed_column in existing:
        return

    embed_lock = _get_column_lock(context, embed_column)
    with embed_lock:
        # Re-check within the lock
        existing = unify.get_fields(context=context)
        if embed_column in existing:
            return
        # Define the embedding equation, with explicit lg scoping
        embed_expr = f"embed({{lg:{source_column}}}, model='{EMBED_MODEL}')"

        url = f"{os.environ['UNIFY_BASE_URL']}/logs/derived"
        headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
        json_input = {
            "project": unify.active_project(),
            "context": context,
            "key": embed_column,
            "equation": embed_expr,
            "referenced_logs": {"lg": {"context": context}},
        }
        response = requests.request("POST", url, json=json_input, headers=headers)
        if response.status_code != 200:
            body = getattr(response, "text", "") or ""
            if (
                "already exists" in body
                or "duplicate key value violates unique constraint" in body
            ):
                return
            assert response.status_code == 200, response.text
        return response.json()
