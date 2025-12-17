"""
Docling document graph construction (PDF/DOCX): hybrid chunking + fallbacks.

Key features:
- Hybrid chunking (Docling HybridChunker) when available
- Waterfall fallbacks: hybrid → native docling traversal → markdown/text splitting
- Section hierarchy via Docling heading paths (doc index + chunk meta headings)
- Sentence splitting: spaCy → regex fallback (best-effort)
- Attachment of merged Docling tables and images into the graph
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from unity.common.token_utils import clip_text_to_token_limit_conservative
from unity.file_manager.file_parsers.settings import FileParserSettings
from unity.file_manager.file_parsers.implementations.docling.types.structure_index import (
    DoclingStructureIndex,
)
from unity.file_manager.file_parsers.types.graph import (
    BBox,
    ContentGraph,
    ContentNode,
    DocumentPayload,
    ImagePayload,
    ParagraphPayload,
    Provenance,
    SectionPayload,
    SentencePayload,
    TablePayload,
)
from unity.file_manager.file_parsers.types.table import ExtractedTable

from .docling_graph import build_document_graph_from_docling


def _bbox_from_docling(bbox_obj: Any) -> Optional[BBox]:
    if bbox_obj is None:
        return None
    d: Dict[str, Any] = {}
    try:
        if hasattr(bbox_obj, "model_dump"):
            d = bbox_obj.model_dump()
        else:
            d = {k: getattr(bbox_obj, k, None) for k in ("l", "t", "r", "b")}
    except Exception:
        d = {}

    try:
        origin_val = getattr(bbox_obj, "coord_origin", None)
        origin = getattr(origin_val, "value", None) or getattr(origin_val, "name", None)
    except Exception:
        origin = None

    try:
        l = float(d.get("l"))
        t = float(d.get("t"))
        r = float(d.get("r"))
        b = float(d.get("b"))
        return BBox(l=l, t=t, r=r, b=b, origin=origin)
    except Exception:
        return None


def _prov_from_doc_items(doc_items: Sequence[Any]) -> Optional[Provenance]:
    """Best-effort provenance from the first doc_item with prov."""
    for di in list(doc_items or []):
        try:
            prov = getattr(di, "prov", None) or []
            if not prov:
                continue
            p0 = prov[0]
            return Provenance(
                page_no=getattr(p0, "page_no", None),
                bbox=_bbox_from_docling(getattr(p0, "bbox", None)),
                source_ref=getattr(di, "self_ref", None),
            )
        except Exception:
            continue
    return None


@lru_cache(maxsize=2)
def _load_spacy(model_name: str):
    import spacy  # type: ignore

    # Keep pipeline minimal; sentence boundaries are the goal.
    return spacy.load(model_name)


def split_sentences(text: str, *, settings: FileParserSettings) -> List[str]:
    """Split text into sentences with spaCy → regex fallback."""
    import re

    text_norm = re.sub(r"\s+", " ", (text or "").strip())
    if not text_norm:
        return []

    # 1) spaCy
    try:
        nlp = _load_spacy(settings.SPACY_MODEL)
        # Apply optional sentence-boundary fixes (enumeration-only fragments).
        from unity.file_manager.file_parsers.utils.spacy_utils import (
            ensure_sentence_fixes,
        )

        nlp = ensure_sentence_fixes(nlp)
        doc = nlp(text_norm)
        sents = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]
        if sents:
            return sents
    except Exception:
        pass

    # 2) Regex fallback
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"])", text_norm)
    cleaned: List[str] = []
    for s in chunks:
        s2 = re.sub(r"^[\s\-–—•·]*([.!?])+\s*", "", s).strip()
        if s2:
            cleaned.append(s2)
    return cleaned


def _is_likely_header(text: str, *, doc_items: Sequence[Any] | None = None) -> bool:
    """Heuristic header detection (ported from legacy DoclingParser, simplified)."""
    import re

    t = (text or "").strip()
    if not t:
        return False
    if len(t) > 200:
        return False

    tl = t.lower()
    section_keywords = [
        "introduction",
        "conclusion",
        "abstract",
        "summary",
        "overview",
        "background",
        "methodology",
        "methods",
        "results",
        "discussion",
        "references",
        "bibliography",
        "appendix",
        "chapter",
        "section",
        "part",
        "contents",
        "preface",
        "acknowledgments",
        "foreword",
        "executive summary",
        "table of contents",
        "list of figures",
    ]
    if any(tl.startswith(k) for k in section_keywords):
        return True

    if t.isupper() and len(t.split()) < 15:
        return True

    if len(t) < 60 and not t.endswith((".", ",", ";", ":", "!", "?", '"', "'")):
        if t[0].isupper():
            return True

    if t.istitle():
        return True

    if t.startswith(("#", "##", "###")):
        return True

    if re.match(r"^[\d\.]+\s+", t):
        return True
    if re.match(r"^[A-Z]\.\s+", t):
        return True
    if re.match(r"^\([a-zA-Z0-9]+\)\s+", t):
        return True
    if re.match(r"^[IVXLCDM]+\.?\s+", t):
        return True
    if re.match(r"^(Chapter|Section|Part)\s+[\dIVXLCDM]+", t, re.I):
        return True

    for di in list(doc_items or []):
        try:
            lab = str(getattr(di, "label", "") or "").lower()
            if lab in {"title", "heading", "section_header"} or lab.startswith("h"):
                return True
        except Exception:
            continue
    return False


def _headings_for_chunk(
    chunk: Any,
    *,
    doc_index: Optional[DoclingStructureIndex],
) -> List[str]:
    """Derive a heading path for a chunk from meta.headings and doc_index ref_to_path."""
    headings: List[str] = []
    try:
        meta = getattr(chunk, "meta", None)
        hs = getattr(meta, "headings", None) if meta is not None else None
        if hs:
            headings = [str(h).strip() for h in list(hs) if str(h).strip()]
    except Exception:
        headings = []

    # Strengthen/replace using doc_index if doc_items are present.
    try:
        if doc_index and hasattr(getattr(chunk, "meta", None), "doc_items"):
            doc_items = list(getattr(chunk.meta, "doc_items", []) or [])
            paths: list[tuple[str, ...]] = []
            for di in doc_items:
                ref = getattr(di, "self_ref", None)
                ref_key = str(ref) if ref is not None else ""
                if ref_key and ref_key in doc_index.ref_to_path:
                    paths.append(tuple(doc_index.ref_to_path[ref_key]))
            if paths:
                common_path, _count = Counter(paths).most_common(1)[0]
                headings = [str(x) for x in list(common_path) if str(x).strip()]
    except Exception:
        pass

    return headings


class _IdGen:
    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)

    def next(self, kind: str) -> str:
        i = self._counters[kind]
        self._counters[kind] = i + 1
        return f"{kind}:{i}"


class _GraphBuilder:
    def __init__(self) -> None:
        self.ids = _IdGen()
        self.nodes: Dict[str, ContentNode] = {}
        self.root_id: Optional[str] = None
        self._section_id_by_path: Dict[Tuple[str, ...], str] = {}
        self._section_order = 0
        self._paragraph_order = 0

    def add_node(
        self,
        *,
        kind: str,
        parent_id: Optional[str],
        title: Optional[str] = None,
        text: Optional[str] = None,
        summary: Optional[str] = None,
        payload: Any = None,
        provenance: Optional[Provenance] = None,
        meta: Optional[Dict[str, Any]] = None,
        order: Optional[int] = None,
    ) -> str:
        node_id = self.ids.next(kind)
        node = ContentNode(
            node_id=node_id,
            kind=kind,  # type: ignore[arg-type]
            parent_id=parent_id,
            children_ids=[],
            order=order,
            title=title,
            text=text,
            summary=summary,
            provenance=provenance,
            payload=payload,
            meta=dict(meta or {}),
        )
        self.nodes[node_id] = node
        if parent_id is not None and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node_id)
        return node_id

    def ensure_section_path(self, path: Sequence[str]) -> str:
        """Ensure nested section nodes exist for the full path and return deepest section id."""
        assert self.root_id is not None
        if not path:
            key = ("Document Content",)
        else:
            key = tuple(str(x).strip() for x in path if str(x).strip())
            if not key:
                key = ("Document Content",)

        # Create intermediate nodes
        cur_parent = self.root_id
        for depth in range(1, len(key) + 1):
            sub = key[:depth]
            if sub in self._section_id_by_path:
                cur_parent = self._section_id_by_path[sub]
                continue
            self._section_order += 1
            sec_id = self.add_node(
                kind="section",
                parent_id=cur_parent,
                title=sub[-1],
                payload=SectionPayload(level=depth, path=list(sub)),
                meta={"path": list(sub), "level": depth},
                order=self._section_order,
            )
            self._section_id_by_path[sub] = sec_id
            cur_parent = sec_id
        return cur_parent

    def add_paragraph(
        self,
        *,
        section_id: str,
        text: str,
        headings: Sequence[str],
        chunk_index: int,
        provenance: Optional[Provenance],
        settings: FileParserSettings,
    ) -> str:
        self._paragraph_order += 1
        clipped = clip_text_to_token_limit_conservative(
            (text or "").strip(),
            settings.EMBEDDING_MAX_INPUT_TOKENS
            * 4,  # allow paragraph text > embedding budget (summary will be clipped)
            settings.EMBEDDING_ENCODING,
        )
        return self.add_node(
            kind="paragraph",
            parent_id=section_id,
            text=clipped,
            payload=ParagraphPayload(),
            provenance=provenance,
            meta={
                "chunk_index": int(chunk_index),
                "headings_path": [str(x) for x in list(headings or [])],
            },
            order=self._paragraph_order,
        )

    def add_sentence_nodes(
        self,
        *,
        paragraph_id: str,
        sentences: Sequence[str],
        provenance: Optional[Provenance],
    ) -> None:
        sent_ix = 0
        for s in list(sentences or []):
            st = (s or "").strip()
            if not st:
                continue
            sent_ix += 1
            self.add_node(
                kind="sentence",
                parent_id=paragraph_id,
                text=st,
                payload=SentencePayload(sentence_index=sent_ix),
                provenance=provenance,
                order=sent_ix,
            )

    def finish(self) -> ContentGraph:
        assert self.root_id is not None, "root_id must be set"
        return ContentGraph(root_id=self.root_id, nodes=self.nodes)


def _try_new_hybrid_chunker(*, settings: FileParserSettings):
    """Return a Docling HybridChunker instance, or raise if unavailable."""
    try:
        from docling.chunking import HybridChunker  # type: ignore
        from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer  # type: ignore
        import tiktoken  # type: ignore
    except Exception as e:
        raise RuntimeError("Hybrid chunking dependencies unavailable") from e

    enc = tiktoken.encoding_for_model(settings.EMBEDDING_MODEL)
    tokenizer = OpenAITokenizer(
        tokenizer=enc,
        max_tokens=settings.EMBEDDING_MAX_INPUT_TOKENS,
    )
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


@dataclass(frozen=True)
class BuiltDocumentGraph:
    graph: ContentGraph
    tables: List[ExtractedTable]
    strategy: str


def build_document_graph_hybrid(
    docling_doc: Any,
    *,
    doc_index: Optional[DoclingStructureIndex],
    merged_table_items: Optional[Sequence[object]] = None,
    settings: FileParserSettings,
) -> BuiltDocumentGraph:
    """Build a document ContentGraph using Docling HybridChunker (preferred)."""
    chunker = _try_new_hybrid_chunker(settings=settings)
    chunks = list(chunker.chunk(docling_doc))

    b = _GraphBuilder()
    doc_id = b.add_node(kind="document", parent_id=None, payload=DocumentPayload())
    b.root_id = doc_id

    # Title from doc_index when available
    try:
        title = (
            str(getattr(doc_index, "title", None) or "").strip() if doc_index else ""
        )
        if title:
            b.nodes[doc_id].title = title
    except Exception:
        pass

    prev_headings: List[str] = []

    for i, chunk in enumerate(chunks):
        raw = str(getattr(chunk, "text", "") or "").strip()
        if not raw:
            continue

        headings = _headings_for_chunk(chunk, doc_index=doc_index) or prev_headings
        if headings:
            prev_headings = list(headings)

        # Heading detection: if chunk is a pure heading line, do not duplicate as paragraph.
        try:
            doc_items = list(
                getattr(getattr(chunk, "meta", None), "doc_items", []) or [],
            )
        except Exception:
            doc_items = []
        is_heading = False
        if headings and raw.strip() == str(headings[-1]).strip():
            is_heading = True
        elif _is_likely_header(raw, doc_items=doc_items):
            is_heading = True

        sec_id = b.ensure_section_path(headings)

        if is_heading and headings and raw.strip() == str(headings[-1]).strip():
            continue

        prov = _prov_from_doc_items(doc_items)
        par_id = b.add_paragraph(
            section_id=sec_id,
            text=raw,
            headings=headings,
            chunk_index=i,
            provenance=prov,
            settings=settings,
        )

        sents = split_sentences(raw, settings=settings)
        b.add_sentence_nodes(paragraph_id=par_id, sentences=sents, provenance=prov)

    # Attach merged tables
    tables_out: List[ExtractedTable] = []
    used_labels: Dict[str, int] = defaultdict(int)

    if merged_table_items is None:
        try:
            merged_table_items = list(getattr(docling_doc, "tables", []) or [])
        except Exception:
            merged_table_items = []

    for idx, tbl in enumerate(list(merged_table_items or []), start=1):
        try:
            ref = getattr(tbl, "self_ref", None)
        except Exception:
            ref = None
        ref_key = str(ref) if ref is not None else None

        path = []
        try:
            if doc_index and ref_key:
                p = doc_index.ref_to_path.get(ref_key)
                if p:
                    path = list(p)
        except Exception:
            path = []

        sec_id = b.ensure_section_path(path)

        # Extract df -> rows/cols
        try:
            df = tbl.export_to_dataframe(doc=docling_doc)
            columns = [str(c) for c in list(df.columns)]
            rows = df.to_dict(orient="records")
        except Exception:
            columns = []
            rows = []
        sample_rows = rows[:25] if rows else []

        base_label = (
            " / ".join([str(x) for x in path if str(x).strip()])
            if path
            else f"{idx:02d}"
        )
        if not base_label:
            base_label = f"{idx:02d}"
        used_labels[base_label] += 1
        label = (
            base_label
            if used_labels[base_label] == 1
            else f"{base_label}_{used_labels[base_label]:02d}"
        )

        node_id = b.add_node(
            kind="table",
            parent_id=sec_id,
            title=label,
            provenance=_prov_from_doc_items([tbl]),
            payload=TablePayload(
                label=label,
                columns=columns,
                sample_rows=sample_rows,
                num_rows=len(rows) if rows is not None else None,
                num_cols=len(columns) if columns is not None else None,
            ),
            meta={"table_label": label, "section_path": path},
        )
        tables_out.append(
            ExtractedTable(
                table_id=node_id,
                label=label,
                sheet_name=None,
                columns=columns,
                rows=rows,
                sample_rows=sample_rows,
                num_rows=len(rows) if rows is not None else None,
                num_cols=len(columns) if columns is not None else None,
            ),
        )

    # Attach images (best-effort)
    try:
        from docling_core.types.doc.labels import DocItemLabel  # type: ignore

        for item, _level in docling_doc.iterate_items(
            with_groups=False,
            traverse_pictures=True,
        ):
            try:
                if getattr(item, "label", None) != DocItemLabel.PICTURE:
                    continue
            except Exception:
                continue

            try:
                ref = getattr(item, "self_ref", None)
            except Exception:
                ref = None
            ref_key = str(ref) if ref is not None else None
            path = []
            try:
                if doc_index and ref_key:
                    p = doc_index.ref_to_path.get(ref_key)
                    if p:
                        path = list(p)
            except Exception:
                path = []
            sec_id = b.ensure_section_path(path)

            caption = ""
            try:
                caption = (item.caption_text(docling_doc) or "").strip()
            except Exception:
                caption = ""
            b.add_node(
                kind="image",
                parent_id=sec_id,
                payload=ImagePayload(caption=caption, image_ref=ref_key),
                provenance=_prov_from_doc_items([item]),
                meta=(
                    {"caption": caption, "section_path": path}
                    if caption or path
                    else {}
                ),
            )
    except Exception:
        pass

    return BuiltDocumentGraph(graph=b.finish(), tables=tables_out, strategy="hybrid")


def build_document_graph_fallback(
    docling_doc: Any,
    *,
    doc_index: Optional[DoclingStructureIndex],
    keep_table_self_refs: Optional[set[str]],
    settings: FileParserSettings,
) -> BuiltDocumentGraph:
    """Fallback to native docling traversal graph builder (no hybrid chunking)."""
    built = build_document_graph_from_docling(
        docling_doc,
        keep_table_self_refs=keep_table_self_refs,
    )
    # Ensure title from doc_index if available
    try:
        title = (
            str(getattr(doc_index, "title", None) or "").strip() if doc_index else ""
        )
        if title and built.graph.root_id in built.graph.nodes:
            built.graph.nodes[built.graph.root_id].title = title
    except Exception:
        pass
    return BuiltDocumentGraph(graph=built.graph, tables=built.tables, strategy="native")


def build_document_graph_from_text(
    text: str,
    *,
    settings: FileParserSettings,
) -> BuiltDocumentGraph:
    """Last-resort graph build from plain text (single section, paragraph/sentence split)."""
    b = _GraphBuilder()
    doc_id = b.add_node(kind="document", parent_id=None, payload=DocumentPayload())
    b.root_id = doc_id

    sec_id = b.ensure_section_path(["Document Content"])

    # Split into paragraphs by blank lines
    paras = [p.strip() for p in (text or "").split("\n\n") if p and p.strip()]
    for i, p in enumerate(paras):
        par_id = b.add_paragraph(
            section_id=sec_id,
            text=p,
            headings=["Document Content"],
            chunk_index=i,
            provenance=None,
            settings=settings,
        )
        b.add_sentence_nodes(
            paragraph_id=par_id,
            sentences=split_sentences(p, settings=settings),
            provenance=None,
        )

    return BuiltDocumentGraph(graph=b.finish(), tables=[], strategy="text")
