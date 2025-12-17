"""
Docling structure indexing helpers.

These helpers build a lightweight "index" from a DoclingDocument that maps:
- item.self_ref -> section heading path (tuple[str, ...])

This enables:
- consistent attachment of tables/images to the correct section path
- stable table merging heuristics (merge only when section_path matches)
"""

from __future__ import annotations

from typing import Tuple

from unity.file_manager.file_parsers.implementations.docling.types.structure_index import (
    DoclingHeadingOrderItem,
    DoclingStructureIndex,
)


def index_docling_structure(docling_doc: object) -> DoclingStructureIndex:
    """Build an index of headings and item → section-path mapping from Docling's tree.

    Returns a dict with keys:
    - ref_to_path: map of NodeItem.self_ref -> tuple[str, ...] headings path
    - heading_refs: set of self_refs that are SectionHeaderItems (or Title)
    - heading_ref_to_level: map of heading self_ref -> explicit Docling level (int)
    - heading_order: list of {text, level, self_ref, path} in document order
    - title: optional top-level title text if present
    """
    ref_to_path: dict[str, Tuple[str, ...]] = {}
    heading_refs: set[str] = set()
    heading_ref_to_level: dict[str, int] = {}
    heading_order: list[DoclingHeadingOrderItem] = []
    title: str | None = None

    # Maintain a stack of (text, level, self_ref) for section headers
    heading_stack: list[tuple[str, int, str]] = []

    try:
        iterator = docling_doc.iterate_items(  # type: ignore[attr-defined]
            with_groups=True,
            traverse_pictures=False,
        )
    except Exception:
        # Fallback: no index available
        return DoclingStructureIndex()

    for item, _depth in iterator:
        try:
            # Determine label string
            label_str = str(getattr(item, "label", "")).lower()
            self_ref = getattr(item, "self_ref", None)
            text_val = (getattr(item, "text", "") or "").strip()

            # Is this a heading-like node?
            is_section_header = "section_header" in label_str
            is_title = label_str.endswith("title") or label_str == "title"

            if is_section_header or is_title:
                # Prefer Docling's explicit level
                raw_level = getattr(item, "level", None)
                header_level = int(raw_level) if isinstance(raw_level, int) else 1

                # Maintain a proper stack (levels are 1-based)
                while heading_stack and heading_stack[-1][1] >= header_level:
                    heading_stack.pop()
                if self_ref is None:
                    # If no self_ref, fabricate a stable key from text (rare)
                    self_ref = f"#/_heading/{len(heading_order)}"

                heading_stack.append((text_val or "Untitled", header_level, self_ref))

                # Current path
                path: Tuple[str, ...] = tuple(h[0] for h in heading_stack)

                heading_refs.add(self_ref)
                heading_ref_to_level[self_ref] = header_level
                ref_to_path[self_ref] = path
                heading_order.append(
                    DoclingHeadingOrderItem(
                        text=text_val,
                        level=header_level,
                        self_ref=self_ref,
                        path=list(path),
                    ),
                )

                if is_title and text_val and not title:
                    title = text_val

            # Map every item's self_ref to the current path (if any)
            if self_ref is not None and heading_stack:
                ref_to_path[str(self_ref)] = tuple(h[0] for h in heading_stack)
        except Exception:
            continue

    return DoclingStructureIndex(
        ref_to_path=ref_to_path,
        heading_refs=heading_refs,
        heading_ref_to_level=heading_ref_to_level,
        heading_order=heading_order,
        title=title,
    )
