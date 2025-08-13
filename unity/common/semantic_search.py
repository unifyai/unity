from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import hashlib
import os

import requests
import unify

from .embed_utils import EMBED_MODEL, ensure_vector_column
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


def ensure_sum_cosine_column(
    context: str,
    terms: List[Tuple[str, str]],
    seed: str,
) -> str:
    """Create a summed cosine derived column over provided (embed_col, ref_text) terms.

    Returns the created (or existing) sum column key.
    """
    # Build terms with lg alias
    eq_terms: list[str] = []
    for embed_col, ref_text in terms:
        escaped_ref = escape_single_quotes(ref_text)
        eq_terms.append(
            f"cosine({{lg:{embed_col}}}, embed('{escaped_ref}', model='{EMBED_MODEL}'))",
        )

    sum_key = f"_sum_cos_{seed}"
    sum_equation = " + ".join(eq_terms) if eq_terms else "0"

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


def fetch_top_k_by_terms(
    context: str,
    terms: List[Tuple[str, str]],
    *,
    k: int = 5,
    row_filter: Optional[str] = None,
) -> List[dict]:
    """Return top-k rows ranked by semantic similarity given pre-embedded terms.

    terms is a list of (embed_column_name, reference_text) pairs that already exist
    in the provided context.
    """
    exclude_fields = [
        fld for fld in unify.get_fields(context=context).keys() if fld.endswith("_emb")
    ]

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
            exclude_fields=exclude_fields,
        )
        return [lg.entries for lg in logs]

    canonical = "|".join(f"{i}:{col}=>{txt}" for i, (col, txt) in enumerate(terms))
    import hashlib as _hashlib

    sum_hash = _hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]
    sum_key = ensure_sum_cosine_column(context, terms, sum_hash)

    logs = unify.get_logs(
        context=context,
        filter=row_filter,
        sorting={sum_key: "ascending"},
        limit=k,
        exclude_fields=exclude_fields,
    )
    return [lg.entries for lg in logs]


def fetch_top_k_by_references(
    context: str,
    references: Dict[str, str],
    *,
    k: int = 5,
    row_filter: Optional[str] = None,
) -> List[dict]:
    """Return top-k rows from a context ranked by semantic similarity to reference text(s).

    This helper abstracts the common flow used by KnowledgeManager's semantic search methods:
    - Ensure an embedding column exists for each source expression (plain column or derived expr)
    - Rank by cosine when a single source is provided
    - Rank by the sum of cosine distances across multiple sources when more than one is provided
    - Exclude embedding columns ("*_emb") from the result payloads
    """
    assert (
        isinstance(references, dict) and len(references) > 0
    ), "references must be a non-empty dict"

    # Collect (embed_col, ref_text) pairs
    terms: List[Tuple[str, str]] = []
    for source_expr, ref_text in references.items():
        embed_col = ensure_vector_for_source(context, source_expr)
        terms.append((embed_col, ref_text))

    return fetch_top_k_by_terms(context, terms, k=k, row_filter=row_filter)
