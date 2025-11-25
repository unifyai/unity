from __future__ import annotations

from typing import Dict, List, Optional


def build_parent_chain(
    auto_counting: Optional[Dict[str, Optional[str]]],
    id_key: str,
) -> List[str]:
    """Return the list of ancestor id keys for a given id_key using an auto_counting map.

    The chain is ordered from nearest parent outward, e.g. for 'sentence_id' →
    ['paragraph_id', 'section_id', 'document_id'] when configured.
    """
    chain: List[str] = []
    if not auto_counting:
        return chain
    current = id_key
    while True:
        parent = auto_counting.get(current)
        if parent is None:
            break
        chain.append(parent)
        current = parent
    return chain


def set_row_ids(
    row: Dict[str, object],
    id_key_for_row: Optional[str],
    *,
    auto_counting: Optional[Dict[str, Optional[str]]],
    document_index: Optional[int] = None,
    section_index: Optional[int] = None,
    paragraph_index: Optional[int] = None,
) -> None:
    """Populate hierarchical id fields on a flattened row according to parentage.

    This mirrors the helper previously nested inside Document.to_schema_rows and is
    shared so both the parser and document utilities can remain consistent.
    """
    if not id_key_for_row:
        return
    for key in build_parent_chain(auto_counting, id_key_for_row):
        if key == "document_id" and document_index is not None:
            row[key] = int(document_index)
        elif key == "section_id" and section_index is not None:
            row[key] = int(section_index)
        elif key == "paragraph_id" and paragraph_index is not None:
            row[key] = int(paragraph_index)
