"""Concurrent ``search`` on a freshly-shared Knowledge table converges.

The contract documented on :meth:`KnowledgeManager.search`:

- Each call may synchronously mutate the Knowledge schema and backfill
  per-source embeddings on its first pass against a reference column.
- When two bodies race into that first call the DB resolves the
  collision through idempotent inserts and unique-constraint guards.
- No distributed lock exists at the Unity layer; the test pins that
  two concurrent calls both succeed without raising, and that the
  final row set is identical across both callers.

The test drives the live backend because the race is by definition
DB-resolved — a mock would erase the behavior under test.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import pytest

from tests.helpers import _handle_project
from unity.knowledge_manager.knowledge_manager import KnowledgeManager


_TABLE = "ConcurrentSearch"
_ROWS: List[Dict[str, str]] = [
    {"content": "I prefer e-mail for follow-ups.", "channel": "email"},
    {"content": "Text messaging is my go-to for quick replies.", "channel": "sms"},
    {"content": "I love taking the train to work.", "channel": "travel"},
]


def _normalize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop fields that vary per call so the two result sets can be compared.

    Semantic-distance scores depend on the embedding call ordering and
    are not part of the convergence contract — the contract is that the
    *row set* converges, not the exact scores.
    """
    stripped: List[Dict[str, Any]] = []
    for row in rows:
        clean = {k: v for k, v in row.items() if k not in {"distance", "_score"}}
        stripped.append(clean)
    return stripped


@pytest.mark.requires_real_unify
@_handle_project
def test_concurrent_search_on_fresh_column_converges():
    km = KnowledgeManager()
    km._create_table(
        name=_TABLE,
        columns={"content": "str", "channel": "str"},
    )
    km._add_rows(table=_TABLE, rows=_ROWS)

    results: Dict[int, List[Dict[str, Any]]] = {}
    errors: Dict[int, BaseException] = {}

    def _search_once(idx: int) -> None:
        try:
            results[idx] = km._search(
                table=_TABLE,
                references={"content": "favorite means of communication"},
                k=2,
            )
        except BaseException as exc:
            errors[idx] = exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_search_once, i) for i in range(2)]
        for fut in as_completed(futures):
            fut.result()

    assert not errors, f"concurrent search() raised: {errors}"
    assert set(results) == {0, 1}

    left = _normalize(results[0])
    right = _normalize(results[1])
    assert left and right
    assert len(left) == len(right) == 2
    assert {row["content"] for row in left} == {row["content"] for row in right}
