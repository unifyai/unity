"""
Docling document enrichment (PDF/DOCX): hierarchical summaries + metadata extraction.

This ports the proven behaviors from the legacy `DoclingParser`:
- Paragraph → Section → Document summaries (token-aware, embedding-safe)
- Metadata extraction with retries/backoff (best-effort)

The functions in this module mutate the `ContentGraph` in-place.
"""

from __future__ import annotations

from typing import List, Optional

from unity.common.llm_client import new_llm_client
from unity.common.token_utils import (
    clip_text_to_token_limit_conservative,
    conservative_token_estimate,
    has_meaningful_text,
)
from unity.file_manager.file_parsers.settings import FileParserSettings
from unity.file_manager.file_parsers.types.enums import NodeKind
from unity.file_manager.file_parsers.types.graph import ContentGraph, ContentNode
from unity.file_manager.file_parsers.types.contracts import FileParseMetadata


def _stable_children(graph: ContentGraph, node: ContentNode) -> List[ContentNode]:
    kids = [graph.nodes[cid] for cid in (node.children_ids or []) if cid in graph.nodes]
    kids.sort(
        key=lambda n: ((n.order if n.order is not None else 1_000_000_000), n.node_id),
    )
    return kids


def _iter_nodes_by_kind(graph: ContentGraph, kind: NodeKind) -> List[ContentNode]:
    return [n for n in graph.nodes.values() if n.kind == kind]


def _clip_for_summary(prompt: str, text: str, *, settings: FileParserSettings) -> str:
    # Conservative budgeting: reserve prompt tokens, leave a minimum usable budget.
    prompt_tokens = conservative_token_estimate(prompt, settings.SUMMARY_ENCODING)
    usable = max(int(settings.SUMMARY_MAX_TOKENS) - int(prompt_tokens), 256)
    return clip_text_to_token_limit_conservative(
        text or "",
        usable,
        settings.SUMMARY_ENCODING,
    )


def _generate_embedding_safe_summary(
    *,
    prompt: str,
    source_text: str,
    settings: FileParserSettings,
) -> str:
    """Generate a summary clipped to the embedding budget, with safe compression retries."""
    from unity.file_manager.file_parsers.utils.summary_compression import (
        generate_summary_with_compression,
    )

    client = new_llm_client(
        settings.SUMMARY_MODEL,
        async_client=False,
        reasoning_effort=None,
        service_tier=None,
        debug_marker="FileParser.generate_summary",
    )

    clipped_src = _clip_for_summary(prompt, source_text, settings=settings)
    summary = generate_summary_with_compression(
        client,
        prompt,
        clipped_src,
        embedding_encoding=settings.EMBEDDING_ENCODING,
        max_embedding_tokens=settings.EMBEDDING_MAX_INPUT_TOKENS,
    )
    return clip_text_to_token_limit_conservative(
        summary or "",
        settings.EMBEDDING_MAX_INPUT_TOKENS,
        settings.EMBEDDING_ENCODING,
    )


def generate_hierarchical_summaries(
    graph: ContentGraph,
    *,
    settings: FileParserSettings,
) -> None:
    """
    Populate:
    - paragraph.summary
    - section.summary (from paragraph summaries)
    - document.summary (from section summaries)
    """
    from unity.file_manager.file_parsers.prompts.document_prompts import (
        build_chunked_text_summary_prompt,
        build_document_summary_prompt,
        build_paragraph_summary_prompt,
        build_section_summary_prompt,
    )

    # --------------------- Paragraph summaries --------------------- #
    para_prompt = build_paragraph_summary_prompt(
        embedding_budget_tokens=settings.EMBEDDING_MAX_INPUT_TOKENS,
    )
    paragraphs = _iter_nodes_by_kind(graph, NodeKind.PARAGRAPH)
    for p in paragraphs:
        if p.summary:
            continue
        text = (p.text or "").strip()
        if not has_meaningful_text(text):
            p.summary = text.strip()
            continue
        try:
            p.summary = _generate_embedding_safe_summary(
                prompt=para_prompt,
                source_text=text,
                settings=settings,
            )
        except Exception:
            # Fallback: clipped text
            p.summary = clip_text_to_token_limit_conservative(
                text,
                settings.EMBEDDING_MAX_INPUT_TOKENS,
                settings.EMBEDDING_ENCODING,
            )

    # --------------------- Section summaries --------------------- #
    sec_prompt = build_section_summary_prompt(
        embedding_budget_tokens=settings.EMBEDDING_MAX_INPUT_TOKENS,
    )
    sections = _iter_nodes_by_kind(graph, NodeKind.SECTION)
    for s in sections:
        # Collect child paragraph summaries in stable order
        child_paras = [
            c for c in _stable_children(graph, s) if c.kind == NodeKind.PARAGRAPH
        ]
        parts = []
        for cp in child_paras:
            t = (cp.summary or "").strip()
            if t:
                parts.append(t)
        if not parts:
            # If no paragraphs, fall back to title
            s.summary = (s.title or "").strip()
            continue
        joined = "\n\n".join(parts)

        # Map-reduce fallback when the input is too large for a single summariser call.
        try:
            prompt_tokens = conservative_token_estimate(
                sec_prompt,
                settings.SUMMARY_ENCODING,
            )
            usable = max(int(settings.SUMMARY_MAX_TOKENS) - int(prompt_tokens), 256)
        except Exception:
            usable = int(settings.SUMMARY_MAX_TOKENS)

        try:
            if conservative_token_estimate(joined, settings.SUMMARY_ENCODING) > usable:
                # Chunk paragraph summaries into multiple chunked summaries, then summarise again.
                chunks: List[str] = []
                cur: List[str] = []
                cur_tokens = 0
                for it in parts:
                    it = (it or "").strip()
                    if not it:
                        continue
                    it_tokens = conservative_token_estimate(
                        it,
                        settings.SUMMARY_ENCODING,
                    )
                    if cur and (cur_tokens + it_tokens) > usable:
                        chunks.append("\n\n".join(cur))
                        cur = []
                        cur_tokens = 0
                    cur.append(it)
                    cur_tokens += it_tokens
                if cur:
                    chunks.append("\n\n".join(cur))

                chunk_summaries: List[str] = []
                for ix, ctext in enumerate(chunks, start=1):
                    cprompt = build_chunked_text_summary_prompt(
                        ix,
                        len(chunks),
                        embedding_budget_tokens=settings.EMBEDDING_MAX_INPUT_TOKENS,
                    )
                    chunk_summaries.append(
                        _generate_embedding_safe_summary(
                            prompt=cprompt,
                            source_text=ctext,
                            settings=settings,
                        ),
                    )
                joined_chunks = "\n\n".join(
                    [x for x in chunk_summaries if x and x.strip()],
                )
                s.summary = _generate_embedding_safe_summary(
                    prompt=sec_prompt,
                    source_text=joined_chunks,
                    settings=settings,
                )
            else:
                s.summary = _generate_embedding_safe_summary(
                    prompt=sec_prompt,
                    source_text=joined,
                    settings=settings,
                )
        except Exception:
            s.summary = clip_text_to_token_limit_conservative(
                joined,
                settings.EMBEDDING_MAX_INPUT_TOKENS,
                settings.EMBEDDING_ENCODING,
            )

    # --------------------- Document summary --------------------- #
    doc_prompt = build_document_summary_prompt(
        embedding_budget_tokens=settings.EMBEDDING_MAX_INPUT_TOKENS,
    )
    root = graph.nodes.get(graph.root_id)
    if root is None:
        return
    # Gather section summaries in stable traversal order (root children)
    sec_summaries: List[str] = []
    for child in _stable_children(graph, root):
        if child.kind == NodeKind.SECTION:
            t = (child.summary or "").strip()
            if t:
                sec_summaries.append(t)
    if not sec_summaries:
        # Fall back to concatenated paragraph summaries if there are no sections
        sec_summaries = [
            (p.summary or "").strip() for p in paragraphs if (p.summary or "").strip()
        ]
    joined = "\n\n".join(sec_summaries)
    if not joined:
        root.summary = ""
        return

    try:
        prompt_tokens = conservative_token_estimate(
            doc_prompt,
            settings.SUMMARY_ENCODING,
        )
        usable = max(int(settings.SUMMARY_MAX_TOKENS) - int(prompt_tokens), 256)
    except Exception:
        usable = int(settings.SUMMARY_MAX_TOKENS)

    try:
        if conservative_token_estimate(joined, settings.SUMMARY_ENCODING) > usable:
            chunks: List[str] = []
            cur: List[str] = []
            cur_tokens = 0
            for it in sec_summaries:
                it = (it or "").strip()
                if not it:
                    continue
                it_tokens = conservative_token_estimate(it, settings.SUMMARY_ENCODING)
                if cur and (cur_tokens + it_tokens) > usable:
                    chunks.append("\n\n".join(cur))
                    cur = []
                    cur_tokens = 0
                cur.append(it)
                cur_tokens += it_tokens
            if cur:
                chunks.append("\n\n".join(cur))

            chunk_summaries: List[str] = []
            for ix, ctext in enumerate(chunks, start=1):
                cprompt = build_chunked_text_summary_prompt(
                    ix,
                    len(chunks),
                    embedding_budget_tokens=settings.EMBEDDING_MAX_INPUT_TOKENS,
                )
                chunk_summaries.append(
                    _generate_embedding_safe_summary(
                        prompt=cprompt,
                        source_text=ctext,
                        settings=settings,
                    ),
                )
            joined_chunks = "\n\n".join([x for x in chunk_summaries if x and x.strip()])
            root.summary = _generate_embedding_safe_summary(
                prompt=doc_prompt,
                source_text=joined_chunks,
                settings=settings,
            )
        else:
            root.summary = _generate_embedding_safe_summary(
                prompt=doc_prompt,
                source_text=joined,
                settings=settings,
            )
    except Exception:
        root.summary = clip_text_to_token_limit_conservative(
            joined,
            settings.EMBEDDING_MAX_INPUT_TOKENS,
            settings.EMBEDDING_ENCODING,
        )


def extract_metadata(
    *,
    full_text: str,
    settings: FileParserSettings,
) -> Optional[FileParseMetadata]:
    """
    Best-effort metadata extraction (token-aware retries/backoff).

    Returns FileParseMetadata with comma-separated strings.
    """
    from unity.file_manager.file_parsers.prompts.metadata_prompts import (
        build_metadata_extraction_prompt,
    )
    from unity.file_manager.file_parsers.types.metadata_extraction import (
        DocumentMetadataExtraction,
    )

    if not has_meaningful_text(full_text):
        return FileParseMetadata(
            key_topics="",
            named_entities="",
            content_tags="",
            confidence_score=0.0,
        )

    prompt = build_metadata_extraction_prompt(
        schema_json=DocumentMetadataExtraction.model_json_schema(),
    )

    client = new_llm_client(
        settings.SUMMARY_MODEL,
        async_client=False,
        reasoning_effort=None,
        service_tier=None,
        debug_marker="FileParser.extract_metadata",
    )

    # Prefer the middle of the doc for metadata (often contains key definitions),
    # but include some head/tail context when possible.
    budgets = [
        int(settings.SUMMARY_MAX_TOKENS),
        max(int(settings.SUMMARY_MAX_TOKENS) // 2, 4000),
    ]

    for budget in budgets:
        try:
            try:
                prompt_tokens = conservative_token_estimate(
                    prompt,
                    settings.SUMMARY_ENCODING,
                )
                usable = max(int(budget) - int(prompt_tokens), 256)
            except Exception:
                usable = int(budget)
            clipped = clip_text_to_token_limit_conservative(
                full_text,
                usable,
                settings.SUMMARY_ENCODING,
            )
            resp = client.copy().generate(prompt + clipped).strip()
            validated = DocumentMetadataExtraction.model_validate_json(resp)

            key_topics = ", ".join([str(x) for x in list(validated.key_topics or [])])
            tags = ", ".join([str(x) for x in list(validated.content_tags or [])])

            # Flatten named entities to a comma-separated set
            ents: List[str] = []
            for _k, vs in (validated.named_entities or {}).items():
                try:
                    for v in list(vs or []):
                        s = str(v).strip()
                        if s:
                            ents.append(s)
                except Exception:
                    continue
            ents_str = ", ".join(sorted({e for e in ents}))

            return FileParseMetadata(
                key_topics=key_topics,
                named_entities=ents_str,
                content_tags=tags,
                confidence_score=float(validated.confidence_score),
            )
        except Exception as e:
            continue

    return None
