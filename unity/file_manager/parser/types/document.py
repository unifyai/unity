"""
Standardized document model for hierarchical content representation.
Supports document -> section -> paragraph -> sentence hierarchy.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field as PydanticField

from unity.common.token_utils import has_meaningful_text


class DocumentMetadataExtraction(BaseModel):
    """Pydantic model for LLM-based document metadata extraction."""

    document_type: Literal[
        "policy",
        "procedure",
        "guideline",
        "handbook",
        "form",
        "template",
        "other",
    ] = PydanticField(
        description="Type of document - use exact values only",
    )

    category: Literal[
        "safety",
        "hr",
        "finance",
        "operations",
        "maintenance",
        "legal",
        "tenancy",
        "general",
    ] = PydanticField(
        description="Document category - use exact values only",
    )

    key_topics: List[str] = PydanticField(
        description="Main topics/themes (3-8 items, snake_case format)",
        min_length=3,
        max_length=8,
    )

    named_entities: Dict[str, List[str]] = PydanticField(
        description="Extracted entities organized by type",
        default_factory=dict,
    )

    content_tags: List[str] = PydanticField(
        description="Searchable keywords for query matching (5-12 items)",
        min_length=5,
        max_length=12,
    )

    confidence_score: float = PydanticField(
        description="Confidence in extraction accuracy (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )


class DocumentImage(BaseModel):
    """Individual image with metadata."""

    page: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None
    element_type: Optional[str] = None
    annotation: Optional[str] = None
    annotation_provenance: Optional[str] = None
    # Optional hierarchical hint set by parser from Docling refs
    section_path: Optional[List[str]] = None


class DocumentTable(BaseModel):
    """Individual table with metadata."""

    page: Optional[int] = None
    element_type: Optional[str] = None
    html: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None
    # Optional hierarchical hint set by parser from Docling refs
    section_path: Optional[List[str]] = None
    # Optional sheet/tab name for spreadsheet files (CSV/XLSX)
    sheet_name: Optional[str] = None


class DocumentSentence(BaseModel):
    """Individual sentence with metadata."""

    text: str
    sentence_id: str  # Unique ID
    paragraph_id: str  # Parent paragraph ID
    section_id: str  # Parent section ID
    document_id: str  # Parent document ID

    # Position metadata
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    sentence_index: int = 0  # Index within paragraph

    # Additional metadata
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = PydanticField(default_factory=dict)


class DocumentParagraph(BaseModel):
    """Paragraph containing sentences with hierarchical summary."""

    text: str
    paragraph_id: str  # Unique ID
    section_id: str  # Parent section ID
    document_id: str  # Parent document ID

    # Child sentences
    sentences: List[DocumentSentence] = PydanticField(default_factory=list)

    # Position metadata
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    paragraph_index: int = 0  # Index within section

    # Hierarchical content
    summary: Optional[str] = None  # Generated from sentence summaries

    # Additional metadata
    metadata: Dict[str, Any] = PydanticField(default_factory=dict)


class DocumentSection(BaseModel):
    """Section containing paragraphs with hierarchical summary."""

    title: str
    section_id: str  # Unique ID
    document_id: str  # Parent document ID

    # Child paragraphs
    paragraphs: List[DocumentParagraph] = PydanticField(default_factory=list)

    # Position metadata
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    section_index: int = 0  # Index within document
    level: int = 1  # Heading level (1, 2, 3, etc.)

    # Hierarchical content
    content_text: str = ""
    summary: Optional[str] = None  # Generated from paragraph summaries

    # Additional metadata
    metadata: Dict[str, Any] = PydanticField(default_factory=dict)


class DocumentMetadata(BaseModel):
    """Enhanced document metadata with processing information."""

    # Basic info
    title: str
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None

    # Timestamps
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    processed_at: Optional[str] = None

    # Content classification
    document_type: str = "document"
    category: str = "general"
    language: str = "en"

    # Processing metadata
    parser_name: Optional[str] = None
    parser_version: Optional[str] = None
    processing_time: Optional[float] = None

    # Content statistics
    total_pages: int = 0
    total_sections: int = 0
    total_paragraphs: int = 0
    total_sentences: int = 0
    total_characters: int = 0
    total_words: int = 0

    # Enhanced metadata from LLM
    key_topics: List[str] = PydanticField(default_factory=list)
    named_entities: Dict[str, List[str]] = PydanticField(default_factory=dict)
    content_tags: List[str] = PydanticField(default_factory=list)
    confidence_score: float = 1.0

    # Images and attachments (structured)
    images: List[DocumentImage] = PydanticField(default_factory=list)
    tables: List[DocumentTable] = PydanticField(default_factory=list)

    # Additional metadata
    extra_metadata: Dict[str, Any] = PydanticField(default_factory=dict)


class Document(BaseModel):
    """
    Main Document class representing the complete hierarchical structure.
    This is the standardized interface all parsers must fulfill.
    """

    # Core identification
    document_id: str  # Unique ID
    metadata: DocumentMetadata

    # Hierarchical content
    sections: List[DocumentSection] = PydanticField(default_factory=list)

    # Document-level content
    full_text: str = ""
    summary: Optional[str] = None  # Generated from section summaries

    # Processing status
    processing_status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None

    def get_total_sentences(self) -> int:
        """Get total number of sentences in document."""
        return sum(
            len(para.sentences)
            for section in self.sections
            for para in section.paragraphs
        )

    def get_total_paragraphs(self) -> int:
        """Get total number of paragraphs in document."""
        return sum(len(section.paragraphs) for section in self.sections)

    def get_all_sentences(self) -> List[DocumentSentence]:
        """Get flattened list of all sentences."""
        sentences = []
        for section in self.sections:
            for paragraph in section.paragraphs:
                sentences.extend(paragraph.sentences)
        return sentences

    def get_all_paragraphs(self) -> List[DocumentParagraph]:
        """Get flattened list of all paragraphs."""
        paragraphs = []
        for section in self.sections:
            paragraphs.extend(section.paragraphs)
        return paragraphs

    def get_all_sections(self) -> List[DocumentSection]:
        """Get all sections."""
        return self.sections

    def to_plain_text(self) -> str:
        """
        Extract plain text content from the document.

        Returns:
            The full text content of the document, or if empty,
            concatenated text from all sections/paragraphs.
        """
        if self.full_text:
            return self.full_text

        # Fallback: concatenate all section content
        text_parts = []
        for section in self.sections:
            if section.content_text:
                text_parts.append(section.content_text)
            else:
                # Further fallback: concatenate all paragraph text
                para_texts = [para.text for para in section.paragraphs if para.text]
                if para_texts:
                    text_parts.append("\n\n".join(para_texts))

        return "\n\n".join(text_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Convert metadata, ensuring pydantic models are serialized
        meta = self.metadata.model_dump()
        try:
            # Overwrite images/tables with serialized dicts if models are present
            meta["images"] = [
                (img.model_dump() if hasattr(img, "model_dump") else img)
                for img in getattr(self.metadata, "images", [])
            ]
            meta["tables"] = [
                (tbl.model_dump() if hasattr(tbl, "model_dump") else tbl)
                for tbl in getattr(self.metadata, "tables", [])
            ]
        except Exception:
            pass

        return {
            "document_id": self.document_id,
            "metadata": meta,
            "sections": [
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "summary": section.summary,
                    "content_text": section.content_text,
                    "level": section.level,
                    "metadata": section.metadata,
                    "paragraphs": [
                        {
                            "paragraph_id": para.paragraph_id,
                            "text": para.text,
                            "summary": para.summary,
                            "metadata": para.metadata,
                            "sentences": [
                                {
                                    "sentence_id": sent.sentence_id,
                                    "text": sent.text,
                                    "confidence_score": sent.confidence_score,
                                    "metadata": sent.metadata,
                                }
                                for sent in para.sentences
                            ],
                        }
                        for para in section.paragraphs
                    ],
                }
                for section in self.sections
            ],
            "full_text": self.full_text,
            "summary": self.summary,
            "processing_status": self.processing_status,
            "error_message": self.error_message,
        }

    def to_schema_rows(
        self,
        *,
        auto_counting: Dict[str, Optional[str]] | None = None,
        document_index: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert hierarchical document to schema-aligned flat rows with hierarchical IDs
        populated according to the provided auto-counting configuration.

        Rules:
        - Document rows: no IDs provided; server assigns `document_id`.
        - Section rows: include only `document_id`.
        - Paragraph rows: include `document_id` and `section_id`.
        - Sentence rows: include `document_id`, `section_id`, `paragraph_id`.
        - Image/Table rows: include `document_id` and `section_id`.

        The `document_index` should be the zero-based index of this document within
        the current ingestion batch, ensuring child rows reference the correct parent.

        Returns:
            List[dict]: rows ready for insertion.
        """
        records: List[Dict[str, Any]] = []

        # Helper to create base record with common fields (schema-aligned only)
        def create_base_record(
            content_type: str,
            title: str,
            summary: Optional[str],
            content_text: str,
        ) -> Dict[str, Any]:
            title = title or f"Untitled {content_type}"
            return {
                "file_path": self.metadata.file_path or "",
                "content_type": content_type,
                "title": title,
                "summary": summary,
                "content_text": content_text or title,
            }

        # Helper: determine the parent chain (excluding the id itself) for an id key
        # e.g. 'sentence_id' -> ['paragraph_id', 'section_id', 'document_id']
        def parent_chain(id_key: str) -> List[str]:
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

        # Helper: set hierarchy IDs on a row based on the id key for this row
        def set_ids(
            row: Dict[str, Any],
            id_key_for_row: Optional[str],
            *,
            sec_index: Optional[int] = None,
            para_index: Optional[int] = None,
        ) -> None:
            if not id_key_for_row:
                return
            for key in parent_chain(id_key_for_row):
                if key == "document_id" and document_index is not None:
                    row[key] = int(document_index)
                elif key == "section_id" and sec_index is not None:
                    row[key] = int(sec_index)
                elif key == "paragraph_id" and para_index is not None:
                    row[key] = int(para_index)
                # All other keys (e.g. deeper ancestors) will be populated via the loop

        # 1. Document-level row (root of the hierarchy)
        doc_title = (self.metadata.title or "").strip()
        doc_row = create_base_record(
            content_type="document",
            title=doc_title,
            summary=self.summary,
            content_text=self.full_text,
        )
        # No IDs for document-level row
        records.append(doc_row)

        # 2. Section-level rows
        # Helper maps to label images/tables and to compute section indices
        page_to_section_index: Dict[int, int] = {}
        section_id_to_index: Dict[str, int] = {}
        path_to_section_index: Dict[tuple, int] = {}
        for section in self.sections:
            section_row = create_base_record(
                content_type="section",
                title=(
                    f"{doc_title} > {section.title}" if doc_title else section.title
                ),
                summary=section.summary,
                content_text=section.content_text
                or "\n\n".join(p.text for p in section.paragraphs),
            )
            # Include only document_id for section rows
            set_ids(section_row, "section_id", sec_index=section.section_index)
            records.append(section_row)
            section_id_to_index[section.section_id] = section.section_index

            # collect pages from paragraph metadata
            try:
                pages = set()
                for p in section.paragraphs:
                    pn = (
                        p.metadata.get("page_no")
                        if isinstance(p.metadata, dict)
                        else None
                    )
                    if isinstance(pn, int):
                        pages.add(pn)
                for pn in pages:
                    page_to_section_index[pn] = section.section_index
            except Exception:
                pass

            # collect normalized path for matching
            try:
                path = None
                if isinstance(section.metadata, dict):
                    maybe_path = section.metadata.get("path")
                    if isinstance(maybe_path, list) and maybe_path:
                        # normalize to tuple of stripped strings
                        path = tuple(
                            str(x).strip() for x in maybe_path if str(x).strip()
                        )
                if path:
                    path_to_section_index[path] = section.section_index
            except Exception:
                pass

            # 3. Paragraph-level records
            for paragraph in section.paragraphs:
                para_title = f"Paragraph {paragraph.paragraph_index + 1}"
                para_row = create_base_record(
                    content_type="paragraph",
                    title=(
                        f"{doc_title} > {section.title} > {para_title}"
                        if doc_title
                        else f"{section.title} > {para_title}"
                    ),
                    summary=paragraph.summary,
                    content_text=paragraph.text,
                )
                # Include document_id and section_id for paragraphs
                set_ids(
                    para_row,
                    "paragraph_id",
                    sec_index=section.section_index,
                    para_index=paragraph.paragraph_index,
                )
                records.append(para_row)

                # 4. Sentence-level records
                for sentence in paragraph.sentences:
                    sent_title = f"Sentence {sentence.sentence_index + 1}"
                    sent_row = create_base_record(
                        content_type="sentence",
                        title=(
                            f"{doc_title} > {section.title} > {para_title} > {sent_title}"
                            if doc_title
                            else f"{section.title} > {para_title} > {sent_title}"
                        ),
                        summary=(
                            sentence.text if has_meaningful_text(sentence.text) else ""
                        ),  # For sentences, summary == content_text
                        content_text=sentence.text,
                    )
                    # Include document_id, section_id, paragraph_id for sentences
                    set_ids(
                        sent_row,
                        "sentence_id",
                        sec_index=section.section_index,
                        para_index=paragraph.paragraph_index,
                    )
                    records.append(sent_row)

        # 5. Image-level records (children of sections)
        images = getattr(self.metadata, "images", []) or []
        for idx, img in enumerate(images):
            try:
                annotation = (
                    getattr(img, "annotation", None)
                    if hasattr(img, "annotation")
                    else None
                )
                content_text = (annotation or "").strip()
                if not content_text:
                    continue

                page = getattr(img, "page", None) if hasattr(img, "page") else None
                sec_title_for_img = None
                sec_index_for_img: Optional[int] = None
                # a) path-based placement using parser-provided section_path → title
                try:
                    sec_path = getattr(img, "section_path", None)
                    if isinstance(sec_path, list) and sec_path:
                        tuple_path = tuple(
                            str(x).strip() for x in sec_path if str(x).strip()
                        )
                        # try exact, then progressively drop the deepest part
                        while tuple_path and sec_title_for_img is None:
                            if tuple_path in path_to_section_index:
                                sec_index_for_img = path_to_section_index[tuple_path]
                                # We no longer keep section titles map; resolve title lazily
                                matching_sections = [
                                    s
                                    for s in self.sections
                                    if s.section_index == sec_index_for_img
                                ]
                                sec_title_for_img = (
                                    matching_sections[0].title
                                    if matching_sections
                                    else None
                                )
                                break
                            tuple_path = tuple_path[:-1]
                except Exception:
                    pass

                # b) page-based placement → title
                if (
                    sec_index_for_img is None
                    and isinstance(page, int)
                    and page in page_to_section_index
                ):
                    sec_index_for_img = page_to_section_index[page]
                    matching_sections = [
                        s for s in self.sections if s.section_index == sec_index_for_img
                    ]
                    sec_title_for_img = (
                        matching_sections[0].title if matching_sections else None
                    )

                # c) fallback
                if sec_index_for_img is None and self.sections:
                    sec_index_for_img = self.sections[0].section_index
                    sec_title_for_img = self.sections[0].title

                # Content fields
                title = (
                    f"Image {idx + 1}{f' (page {page})' if page is not None else ''}"
                )

                img_row = create_base_record(
                    content_type="image",
                    title=(
                        f"{doc_title} > {sec_title_for_img} > {title}"
                        if doc_title and sec_title_for_img
                        else (
                            f"{sec_title_for_img} > {title}"
                            if sec_title_for_img
                            else title
                        )
                    ),
                    summary=content_text,
                    content_text=content_text,
                )
                # Include document_id and section_id
                set_ids(img_row, "image_id", sec_index=sec_index_for_img)
                # # include a few metadata points
                # img_record["page"] = page
                # img_record["bbox"] = (
                #     getattr(img, "bbox", None) if hasattr(img, "bbox") else None
                # )
                # img_record["element_type"] = (
                #     getattr(img, "element_type", None) if hasattr(img, "element_type") else None
                # )
                records.append(img_row)
            except Exception:
                continue

        # 6. Table-level records (children of sections)
        tables = getattr(self.metadata, "tables", []) or []
        for idx, tbl in enumerate(tables):
            try:
                page = getattr(tbl, "page", None) if hasattr(tbl, "page") else None
                sec_title_for_tbl = None
                sec_index_for_tbl: Optional[int] = None
                # a) path-based placement using parser-provided section_path → title
                try:
                    sec_path = getattr(tbl, "section_path", None)
                    if isinstance(sec_path, list) and sec_path:
                        tuple_path = tuple(
                            str(x).strip() for x in sec_path if str(x).strip()
                        )
                        while tuple_path and sec_title_for_tbl is None:
                            if tuple_path in path_to_section_index:
                                sec_index_for_tbl = path_to_section_index[tuple_path]
                                matching_sections = [
                                    s
                                    for s in self.sections
                                    if s.section_index == sec_index_for_tbl
                                ]
                                sec_title_for_tbl = (
                                    matching_sections[0].title
                                    if matching_sections
                                    else None
                                )
                                break
                            tuple_path = tuple_path[:-1]
                except Exception:
                    pass

                # b) page-based placement → title
                if (
                    sec_index_for_tbl is None
                    and isinstance(page, int)
                    and page in page_to_section_index
                ):
                    sec_index_for_tbl = page_to_section_index[page]
                    matching_sections = [
                        s for s in self.sections if s.section_index == sec_index_for_tbl
                    ]
                    sec_title_for_tbl = (
                        matching_sections[0].title if matching_sections else None
                    )

                # c) fallback
                if sec_index_for_tbl is None and self.sections:
                    sec_index_for_tbl = self.sections[0].section_index
                    sec_title_for_tbl = self.sections[0].title

                html = getattr(tbl, "html", None) if hasattr(tbl, "html") else None
                sheet_name = (
                    getattr(tbl, "sheet_name", None)
                    if hasattr(tbl, "sheet_name")
                    else None
                )

                # Build title with sheet name if available
                if sheet_name:
                    title = f"Table {idx + 1} - {sheet_name}{f' (page {page})' if page is not None else ''}"
                else:
                    title = f"Table {idx + 1}{f' (page {page})' if page is not None else ''}"
                content_text = (html or "").strip()

                tbl_row = create_base_record(
                    content_type="table",
                    title=(
                        f"{doc_title} > {sec_title_for_tbl} > {title}"
                        if doc_title and sec_title_for_tbl
                        else (
                            f"{sec_title_for_tbl} > {title}"
                            if sec_title_for_tbl
                            else title
                        )
                    ),
                    summary=content_text,
                    content_text=content_text,
                )
                # Include document_id and section_id
                set_ids(tbl_row, "table_id", sec_index=sec_index_for_tbl)
                # tbl_record["page"] = page
                # tbl_record["bbox"] = (
                #     getattr(tbl, "bbox", None) if hasattr(tbl, "bbox") else None
                # )
                # tbl_record["element_type"] = (
                #     getattr(tbl, "element_type", None) if hasattr(tbl, "element_type") else None
                # )
                records.append(tbl_row)
            except Exception:
                continue

        return records

    # Backward-compat wrapper (deprecated)
    def to_flat_records(self) -> List[Dict[str, Any]]:
        return self.to_schema_rows()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create Document from dictionary with nested Pydantic models."""
        # Build metadata with nested models for images/tables if present
        md = data.get("metadata", {})
        images = [DocumentImage(**img) for img in md.get("images", [])]
        tables = [DocumentTable(**tbl) for tbl in md.get("tables", [])]
        metadata = DocumentMetadata(
            **{k: v for k, v in md.items() if k not in ("images", "tables")},
            images=images,
            tables=tables,
        )

        sections: List[DocumentSection] = []
        for section_data in data.get("sections", []):
            paragraphs: List[DocumentParagraph] = []
            for para_data in section_data.get("paragraphs", []):
                sentences = [
                    DocumentSentence(**s) for s in para_data.get("sentences", [])
                ]
                paragraphs.append(
                    DocumentParagraph(
                        text=para_data.get("text", ""),
                        paragraph_id=para_data.get("paragraph_id", ""),
                        section_id=para_data.get("section_id", ""),
                        document_id=para_data.get("document_id", ""),
                        sentences=sentences,
                        summary=para_data.get("summary"),
                        metadata=para_data.get("metadata", {}),
                    ),
                )
            sections.append(
                DocumentSection(
                    title=section_data.get("title", ""),
                    section_id=section_data.get("section_id", ""),
                    document_id=section_data.get("document_id", ""),
                    paragraphs=paragraphs,
                    summary=section_data.get("summary"),
                    content_text=section_data.get("content_text", ""),
                    level=section_data.get("level", 1),
                    metadata=section_data.get("metadata", {}),
                ),
            )

        return cls(
            document_id=data.get("document_id", ""),
            metadata=metadata,
            sections=sections,
            full_text=data.get("full_text", ""),
            summary=data.get("summary"),
            processing_status=data.get("processing_status", "pending"),
            error_message=data.get("error_message"),
        )


def build_metadata_extraction_prompt() -> str:
    """Build prompt for LLM-based metadata extraction using Pydantic model validation."""
    from unity.file_manager.parser.types.document import DocumentMetadataExtraction

    # Get the Pydantic model schema
    schema = DocumentMetadataExtraction.model_json_schema()

    return f"""
DOCUMENT METADATA EXTRACTION FOR MIDLAND HEART RAG SYSTEM

You are an expert document analyzer for Midland Heart, a UK social housing provider.
Extract comprehensive metadata from the provided document text to enable effective RAG retrieval.

RESPONSE FORMAT:
Your response must be a valid JSON object that exactly matches this Pydantic model schema:

{json.dumps(schema, indent=2)}

FIELD GUIDELINES:

1. **document_type**: Choose from the exact literal values only
2. **category**: Choose from the exact literal values only
3. **summary**: 2-3 sentences focusing on what tenants/staff need to know
4. **key_topics**: Use snake_case format (e.g., "mobility_scooters", "fire_safety")
5. **named_entities**: Organize by type:
   - "organizations": ["Midland Heart", "DVLA", etc.]
   - "policies": Referenced policy names
   - "locations": Specific places mentioned
   - "numbers": Important numbers, limits, percentages
   - "dates": Key dates and deadlines
   - "legislation": Laws and regulations referenced
6. **content_tags**: Include synonyms and related search terms
7. **confidence_score**: Your confidence in extraction accuracy (0.0-1.0)

EXAMPLE OUTPUT:
{{
  "document_type": "policy",
  "category": "safety",
  "summary": "Policy governing the use and storage of mobility scooters in Midland Heart properties. Sets speed limits, weight restrictions, and storage requirements to ensure fire safety compliance.",
  "key_topics": ["mobility_scooters", "speed_limits", "weight_restrictions", "storage_rules", "fire_safety", "class_2_vehicles"],
  "named_entities": {{
    "organizations": ["Midland Heart", "DVLA", "Fire Service"],
    "policies": ["Fire Safety Policy", "ASB Policy"],
    "locations": ["communal_areas", "stairwells", "corridors"],
    "numbers": ["4_mph", "8_mph", "150_kg", "230_kg"],
    "dates": ["October_2024"],
    "legislation": ["Road_Traffic_Act", "Fire_Safety_Order"]
  }},
  "content_tags": ["mobility_aid", "electric_scooter", "disability_access", "fire_risk", "storage_guidelines", "speed_control"],
  "confidence_score": 0.95
}}

CRITICAL: Return ONLY the JSON object, no additional text or markdown formatting.

ANALYZE THE FOLLOWING DOCUMENT:
"""
