from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import hashlib
import os

import requests
import unify

from .embed_utils import EMBED_MODEL, ensure_vector_column, list_private_fields
from ..helpers import _handle_exceptions


def is_plain_identifier(expr: str) -> bool:
    return ("{" not in expr) and ("}" not in expr) and any(c.isalpha() for c in expr)


def escape_single_quotes(text: str) -> str:
    return text.replace("'", "\\'")


def wrap_str_placeholders(expr: str) -> str:
    """Wrap str({field}) → ((str({field})) if exists({field}) else '') to avoid 'None'."""
    import re as _re

    pattern = _re.compile(r"str\(\{\s*([a-zA-Z_][\w]*)\s*\}\)")

    def _repl(m: _re.Match[str]) -> str:
        fld = m.group(1)
        return f"((str({{{fld}}})) if exists({{{fld}}}) else '')"

    return pattern.sub(_repl, expr)


def ensure_vector_for_source(context: str, source_expr: str) -> str:
    """Ensure an embedding column exists for source_expr within context and return its name.

    - If source_expr is a plain identifier, use that column directly and create `_{col}_emb`.
    - If source_expr is an expression, create a stable derived source column `_expr_<hash>` and
      then create the embedding for it.
    """
    if is_plain_identifier(source_expr):
        source_column_name = source_expr
        embed_column_name = f"_{source_column_name}_emb"
        ensure_vector_column(
            context,
            embed_column=embed_column_name,
            source_column=source_column_name,
            derived_expr=None,
        )
    else:
        expr_hash = hashlib.sha1(source_expr.encode("utf-8")).hexdigest()[:10]
        source_column_name = f"_expr_{expr_hash}"
        embed_column_name = f"{source_column_name}_emb"
        sanitized_expr = wrap_str_placeholders(source_expr)
        ensure_vector_column(
            context,
            embed_column=embed_column_name,
            source_column=source_column_name,
            derived_expr=sanitized_expr,
        )
    return embed_column_name


def ensure_mean_cosine_column(
    context: str,
    terms: List[Tuple[str, str]],
    seed: str,
) -> str:
    """Create a mean-of-cosines derived column over provided (embed_col, ref_text) terms.

    - Each term contributes its cosine distance when the embedding exists, otherwise 0.
    - The final value is (sum of present cosines) / (count of present cosines).
    - If no term is present for a row, the value is 2 (maximal distance).

    Returns the created (or existing) column key.
    """
    # Build numerator (sum of present cosines) and denominator (count of present terms)
    num_terms: list[str] = []
    den_terms: list[str] = []
    for embed_col, ref_text in terms:
        escaped_ref = escape_single_quotes(ref_text)
        num_terms.append(
            (
                "((cosine({lg:"
                + embed_col
                + "}, "
                + f"embed('{escaped_ref}', model='{EMBED_MODEL}'))) "
                + f"if exists({{lg:{embed_col}}}) else 0)"
            ),
        )
        den_terms.append(f"(1 if exists({{lg:{embed_col}}}) else 0)")

    sum_key = f"_sum_cos_{seed}"
    numerator = " + ".join(num_terms) if num_terms else "0"
    denominator = " + ".join(den_terms) if den_terms else "0"
    # mean if denom>0 else 2
    sum_equation = f"(({numerator}) / ({denominator})) if (({denominator}) > 0) else 2"

    existing_fields = unify.get_fields(context=context)
    if sum_key not in existing_fields:
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/derived"
        headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
        json_input = {
            "project": unify.active_project(),
            "context": context,
            "key": sum_key,
            "equation": sum_equation,
            "referenced_logs": {"lg": {"context": context}},
        }
        resp = requests.request("POST", url, json=json_input, headers=headers)
        _handle_exceptions(resp)

    return sum_key


def ensure_mean_cosine_column_piecewise(
    context: str,
    terms: List[Tuple[str, str]],
    seed: str,
) -> str:
    """Create a mean-of-cosines derived column using piecewise terms.

    This mirrors ``ensure_mean_cosine_column`` but, for debugging purposes, it
    stores each item that would be part of the numerator and denominator in its
    own derived column. The final mean column is then defined as the sum of
    those per-term columns divided by the count sum, matching the original
    semantics:

    - Numerator term i: ``((cosine({lg:<embed_i>}, embed('<ref_i>', model=...))) if exists({lg:<embed_i>}) else 0)``
    - Denominator term i: ``(1 if exists({lg:<embed_i>}) else 0)``
    - Mean: ``(sum(num_i) / sum(den_i)) if sum(den_i) > 0 else 2``

    Returns the created (or existing) mean column key.
    """
    # Prepare per-term equation strings
    num_equations: list[str] = []
    den_equations: list[str] = []
    for embed_col, ref_text in terms:
        escaped_ref = escape_single_quotes(ref_text)
        num_equations.append(
            (
                "((cosine({lg:"
                + embed_col
                + "}, "
                + f"embed('{escaped_ref}', model='{EMBED_MODEL}'))) "
                + f"if exists({{lg:{embed_col}}}) else 0)"
            ),
        )
        den_equations.append(f"(1 if exists({{lg:{embed_col}}}) else 0)")

    # Derive keys for each term. We include an index and the provided seed to stay stable.
    num_keys: list[str] = [f"_num_cos_{i}_{seed}" for i in range(len(num_equations))]
    den_keys: list[str] = [f"_den_has_{i}_{seed}" for i in range(len(den_equations))]

    # Mean key distinct from the non-piecewise variant for clarity/debugging
    sum_key = f"_sum_cos_piecewise_{seed}"

    # Fetch existing fields once for idempotency
    existing_fields = unify.get_fields(context=context)

    # Create per-term numerator columns
    for key, equation in zip(num_keys, num_equations):
        if key not in existing_fields:
            url = f"{os.environ['UNIFY_BASE_URL']}/logs/derived"
            headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
            json_input = {
                "project": unify.active_project(),
                "context": context,
                "key": key,
                "equation": equation,
                "referenced_logs": {"lg": {"context": context}},
                "derived": False,
            }
            resp = requests.request("POST", url, json=json_input, headers=headers)
            _handle_exceptions(resp)

    # Create per-term denominator columns
    for key, equation in zip(den_keys, den_equations):
        if key not in existing_fields:
            url = f"{os.environ['UNIFY_BASE_URL']}/logs/derived"
            headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
            json_input = {
                "project": unify.active_project(),
                "context": context,
                "key": key,
                "equation": equation,
                "referenced_logs": {"lg": {"context": context}},
                "derived": False,
            }
            resp = requests.request("POST", url, json=json_input, headers=headers)
            _handle_exceptions(resp)

    # Build the final mean equation from the per-term columns
    numerator = " + ".join(["{lg:" + k + "}" for k in num_keys]) if num_keys else "0"
    denominator = " + ".join(["{lg:" + k + "}" for k in den_keys]) if den_keys else "0"
    sum_equation = f"(({numerator}) / ({denominator})) if (({denominator}) > 0) else 2"

    # Create the piecewise mean column
    existing_fields = unify.get_fields(context=context)
    if sum_key not in existing_fields:
        url = f"{os.environ['UNIFY_BASE_URL']}/logs/derived"
        headers = {"Authorization": f"Bearer {os.environ.get('UNIFY_KEY')}"}
        json_input = {
            "project": unify.active_project(),
            "context": context,
            "key": sum_key,
            "equation": sum_equation,
            "referenced_logs": {"lg": {"context": context}},
            "derived": False,
        }
        resp = requests.request("POST", url, json=json_input, headers=headers)
        _handle_exceptions(resp)

    unify.delete_fields(
        num_keys + den_keys,
        context=context,
    )
    return sum_key


def fetch_top_k_by_terms(
    context: str,
    terms: List[Tuple[str, str]],
    *,
    k: int = 10,
    row_filter: Optional[str] = None,
) -> List[dict]:
    """Return top-k rows ranked by semantic similarity given pre-embedded terms.

    terms is a list of (embed_column_name, reference_text) pairs that already exist
    in the provided context.
    """

    if len(terms) == 0:
        return []

    if len(terms) == 1:
        embed_col, ref_text = terms[0]
        escaped_ref = ref_text.replace("'", "\\'")
        logs = unify.get_logs(
            context=context,
            filter=row_filter,
            sorting={
                f"cosine({embed_col}, embed('{escaped_ref}', model='{EMBED_MODEL}'))": "ascending",
            },
            limit=k,
            exclude_fields=list_private_fields(context),
        )
        return [lg.entries for lg in logs]

    # If multiple terms are provided but the filter excludes all rows, avoid
    # creating a summed-cosine derived column which can fail on empty contexts.
    try:
        if row_filter is not None:
            any_rows = unify.get_logs(
                context=context,
                filter=row_filter,
                limit=1,
                exclude_fields=list_private_fields(context),
            )
            if not any_rows:
                return []
    except Exception:
        # If introspection fails, fall back to attempting the derived approach
        pass

    canonical = "|".join(f"{i}:{col}=>{txt}" for i, (col, txt) in enumerate(terms))
    import hashlib as _hashlib

    sum_hash = _hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    sum_key = ensure_mean_cosine_column_piecewise(context, terms, sum_hash)

    logs = unify.get_logs(
        context=context,
        filter=row_filter,
        sorting={sum_key: "ascending"},
        limit=k,
        exclude_fields=list_private_fields(context),
    )
    return [lg.entries for lg in logs]


def fetch_top_k_by_references(
    context: str,
    references: Optional[Dict[str, str]],
    *,
    k: int = 10,
    row_filter: Optional[str] = None,
) -> List[dict]:
    """Return top-k rows from a context ranked by semantic similarity to reference text(s).

    This helper abstracts the common flow used by KnowledgeManager's semantic search methods:
    - Ensure an embedding column exists for each source expression (plain column or derived expr)
    - Rank by cosine when a single source is provided
    - Rank by the sum of cosine distances across multiple sources when more than one is provided
    - Exclude embedding columns ("*_emb") from the result payloads
    """
    # When no references are provided, skip semantic search entirely and
    # let the caller's backfill logic drive the result set.
    if not references:
        return []

    # Collect (embed_col, ref_text) pairs
    terms: List[Tuple[str, str]] = []
    for source_expr, ref_text in references.items():
        embed_col = ensure_vector_for_source(context, source_expr)
        terms.append((embed_col, ref_text))

    return fetch_top_k_by_terms(context, terms, k=k, row_filter=row_filter)


def backfill_rows(
    context: str,
    initial_rows: List[dict],
    k: int,
    *,
    row_filter: Optional[str] = None,
    unique_id_field: Optional[str] = None,
) -> List[dict]:
    """Backfill similarity results with additional rows to reach k.

    Parameters
    ----------
    context : str
        Unify context to read fallback rows from.
    initial_rows : list[dict]
        Rows already obtained from semantic similarity (ordered best-first).
    k : int
        Desired total number of rows.
    row_filter : str | None, default None
        Optional row-level predicate to apply while backfilling (e.g., exclude system rows).
    unique_id_field : str | None, default None
        Column used to deduplicate rows when backfilling. When ``None``, attempts to infer
        from the context's ``unique_column_ids``.

    Returns
    -------
    list[dict]
        Up to ``k`` rows with original similarity results first, followed by backfilled rows.
    """
    results: List[dict] = list(initial_rows)
    if len(results) >= k:
        return results

    # Determine unique id column if not supplied
    if unique_id_field is None:
        try:
            ctx_info = unify.get_context(context)
            unique_id_field = ctx_info.get("unique_column_ids")
            if isinstance(unique_id_field, list):
                unique_id_field = unique_id_field[0] if unique_id_field else None
        except Exception:
            unique_id_field = None

    # Track already included ids to avoid duplicates
    seen_ids: set = set()
    if unique_id_field:
        for r in results:
            if unique_id_field in r and r.get(unique_id_field) is not None:
                try:
                    seen_ids.add(int(r.get(unique_id_field)))
                except Exception:
                    seen_ids.add(r.get(unique_id_field))

    # Exclude embedding/vector columns in payloads
    needed = k - len(results)
    offset = 0
    while needed > 0:
        fallback_logs = unify.get_logs(
            context=context,
            filter=row_filter,
            offset=offset,
            limit=k,
            exclude_fields=list_private_fields(context),
        )
        if not fallback_logs:
            break

        for lg in fallback_logs:
            entries = getattr(lg, "entries", lg)
            if not isinstance(entries, dict):
                continue
            uid_val = entries.get(unique_id_field) if unique_id_field else None
            if unique_id_field is not None:
                try:
                    comp_val = int(uid_val) if uid_val is not None else None
                except Exception:
                    comp_val = uid_val
                if comp_val is not None and comp_val in seen_ids:
                    continue
            results.append(entries)
            if unique_id_field is not None and uid_val is not None:
                try:
                    seen_ids.add(int(uid_val))
                except Exception:
                    seen_ids.add(uid_val)
            needed -= 1
            if needed == 0:
                break

        offset += k

    return results
