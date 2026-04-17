from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Sequence

from unity.file_manager.file_parsers.types.backend import BaseFileParserBackend
from unity.file_manager.file_parsers.utils.tracing import traced_step
from unity.file_manager.file_parsers.types.contracts import FileParseRequest
from unity.file_manager.file_parsers.types.enums import NodeKind
from unity.file_manager.file_parsers.types.graph import (
    ContentGraph,
    ContentNode,
    DocumentPayload,
    ParagraphPayload,
    SectionPayload,
    SentencePayload,
)
from unity.file_manager.file_parsers.types.formats import FileFormat
from unity.file_manager.file_parsers.types.contracts import (
    FileParseResult,
    FileParseTrace,
    StepStatus,
)

logger = logging.getLogger(__name__)


def _split_into_paragraphs(text: str) -> list[str]:
    """Paragraph splitting with fallbacks: blank-lines → heuristic line splits."""
    src = (text or "").strip()
    if not src:
        return []
    chunks = src.split("\n\n")
    if len(chunks) <= 2 and len(src) > 1000:
        import re

        alt_chunks = re.split(r"\n(?=[A-Z])", src)
        if len(alt_chunks) > len(chunks) * 2:
            chunks = alt_chunks
    paragraphs: list[str] = []
    for chunk in chunks:
        cleaned = chunk.strip()
        if cleaned and len(cleaned) > 20:
            paragraphs.append(cleaned)
    return paragraphs


def _split_sentences(text: str) -> list[str]:
    import re

    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\\s+", t)
    return [p.strip() for p in parts if p and p.strip()]


def _extract_basic_structure(text: str) -> list[tuple[Optional[str], str]]:
    """Ported section/heading heuristics from legacy DoclingParser._extract_basic_structure."""
    import re

    lines = (text or "").split("\n")
    sections: list[tuple[Optional[str], str]] = []
    current_section_lines: list[str] = []
    current_title: Optional[str] = None

    header_patterns = [
        re.compile(r"^#{1,6}\s+(.+)$"),  # Markdown headers
        re.compile(r"^([A-Z][A-Z\s]+)$"),  # ALL CAPS HEADERS
        re.compile(r"^(\d+\.?\s+[A-Z].+)$"),  # Numbered sections
        re.compile(r"^([A-Z][^.!?]*):?\s*$"),  # Title case followed by colon
    ]

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            current_section_lines.append(line)
            continue

        is_header = False
        header_text: Optional[str] = None

        for pattern in header_patterns:
            match = pattern.match(line_stripped)
            if match:
                potential_header = match.group(1) if match.lastindex else line_stripped
                if len(potential_header) < 100 and not potential_header.endswith("."):
                    is_header = True
                    header_text = potential_header.strip("#").strip()
                    break

        # Also check for underline headers (Markdown-style)
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and (set(next_line) <= {"=", "-"} and len(next_line) >= 3):
                is_header = True
                header_text = line_stripped

        if is_header and header_text:
            if current_section_lines or current_title:
                sections.append((current_title, "\n".join(current_section_lines)))
            current_title = header_text
            current_section_lines = []
        else:
            current_section_lines.append(line)

    if current_section_lines or current_title:
        sections.append((current_title, "\n".join(current_section_lines)))

    if not sections:
        sections = [("Document Content", text)]

    return sections


def _read_text_best_effort(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


class TextBackend(BaseFileParserBackend):
    name = "text_backend"
    # Treat common text-like formats as plain text.
    #
    # Note: `.md` is mapped to FileFormat.TXT by `extension_to_format`, so it is
    # supported implicitly as TXT with a markdown MIME type.
    supported_formats: Sequence[FileFormat] = (FileFormat.TXT,)

    def can_handle(self, fmt: Optional[FileFormat]) -> bool:
        return fmt in self.supported_formats

    def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
        started = time.perf_counter()
        path = Path(ctx.source_local_path).expanduser().resolve()

        trace = FileParseTrace(
            logical_path=str(ctx.logical_path),
            backend=self.name,
            file_format=ctx.file_format,
            mime_type=ctx.mime_type,
            status=StepStatus.SUCCESS,
            source_local_path=str(path),
            parsed_local_path=str(path),
        )

        if not path.exists() or not path.is_file():
            trace.status = StepStatus.FAILED
            trace.duration_ms = (time.perf_counter() - started) * 1000.0
            return FileParseResult(
                logical_path=str(ctx.logical_path),
                status="error",
                error=f"File not found: {path}",
                file_format=ctx.file_format,
                mime_type=ctx.mime_type,
                trace=trace,
            )

        try:
            with traced_step(trace, name="read_text") as step:
                text = _read_text_best_effort(path)
                step.counters["bytes"] = len(text.encode("utf-8", errors="ignore"))

            with traced_step(trace, name="build_text_graph") as step:
                # Build graph: document -> section -> paragraph -> sentence
                doc_id = "document:0"
                nodes: dict[str, ContentNode] = {}
                nodes[doc_id] = ContentNode(
                    node_id=doc_id,
                    kind=NodeKind.DOCUMENT,
                    parent_id=None,
                    children_ids=[],
                    title=path.name,
                    payload=DocumentPayload(),
                )

                sec_ix = 0
                para_ix = 0
                sent_total = 0
                for section_title, section_content in _extract_basic_structure(text):
                    sec_id = f"section:{sec_ix}"
                    sec_ix += 1
                    nodes[doc_id].children_ids.append(sec_id)
                    nodes[sec_id] = ContentNode(
                        node_id=sec_id,
                        kind=NodeKind.SECTION,
                        parent_id=doc_id,
                        children_ids=[],
                        order=sec_ix,
                        title=section_title or f"Section {sec_ix}",
                        payload=SectionPayload(
                            level=1,
                            path=[section_title] if section_title else None,
                        ),
                    )

                    for para in _split_into_paragraphs(section_content):
                        para_id = f"paragraph:{sec_ix}:{para_ix}"
                        para_ix += 1
                        nodes[sec_id].children_ids.append(para_id)
                        nodes[para_id] = ContentNode(
                            node_id=para_id,
                            kind=NodeKind.PARAGRAPH,
                            parent_id=sec_id,
                            children_ids=[],
                            order=para_ix,
                            text=para,
                            payload=ParagraphPayload(),
                        )
                        sent_ix = 0
                        for sent in _split_sentences(para):
                            sent_id = f"sentence:{sec_ix}:{para_ix}:{sent_ix}"
                            sent_ix += 1
                            sent_total += 1
                            nodes[para_id].children_ids.append(sent_id)
                            nodes[sent_id] = ContentNode(
                                node_id=sent_id,
                                kind=NodeKind.SENTENCE,
                                parent_id=para_id,
                                children_ids=[],
                                order=sent_ix,
                                text=sent,
                                payload=SentencePayload(sentence_index=sent_ix),
                            )

                step.counters["sections"] = sec_ix
                step.counters["paragraphs"] = para_ix
                step.counters["sentences"] = sent_total

                graph = ContentGraph(root_id=doc_id, nodes=nodes)

            trace.counters["nodes"] = len(graph.nodes)
            trace.duration_ms = (time.perf_counter() - started) * 1000.0
            return FileParseResult(
                logical_path=str(ctx.logical_path),
                status="success",
                file_format=ctx.file_format,
                mime_type=ctx.mime_type,
                tables=[],
                full_text=text if isinstance(text, str) else "",
                trace=trace,
                graph=graph,
            )

        except Exception as e:
            logger.exception("Text parse failed: %s", e)
            trace.status = StepStatus.FAILED
            trace.warnings.append(str(e))
            trace.duration_ms = (time.perf_counter() - started) * 1000.0
            return FileParseResult(
                logical_path=str(ctx.logical_path),
                status="error",
                error=str(e),
                file_format=ctx.file_format,
                mime_type=ctx.mime_type,
                trace=trace,
            )
