"""
Standardized document model for hierarchical content representation.
Supports document -> section -> paragraph -> sentence hierarchy.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field as PydanticField

from ..token_utils import has_meaningful_text
from ....common.llm_helpers import short_id


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

    def to_flat_records(self) -> List[Dict[str, Any]]:
        """
        Convert hierarchical document to flat records compatible with the unified content schema.

        Returns:
            List of dictionaries where each dict represents a row in the content table.
            Includes document, section, paragraph, and sentence level records.
        """
        import hashlib

        records = []

        # # Get current timestamp for ingestion tracking - ensure it's a string
        # # Format: YYYY-MM-DDTHH:MM:SS.ffffffZ (ISO 8601 with microseconds and Z suffix)
        # ingested_at = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        # Compute document fingerprint if not already set
        doc_fingerprint = None
        if self.metadata.file_path:
            # Use file path and content to generate fingerprint
            content_for_hash = f"{self.metadata.file_path}:{self.full_text}"
            doc_fingerprint = hashlib.sha256(content_for_hash.encode()).hexdigest()

        # Helper to create base record with common fields
        def create_base_record(
            content_id: str,
            content_type: str,
            title: str,
            summary: Optional[str],
            content_text: str,
            level: int,
            parent_id: Optional[str] = None,
            confidence_score: float = 1.0,
        ) -> Dict[str, Any]:
            title = title or f"Untitled {content_type}"
            return {
                # Required fields
                "content_id": content_id,
                "content_type": content_type,
                "title": title,
                "summary": summary,
                "content_text": content_text or title,
                # Hierarchy fields
                "level": level,
                "parent_id": parent_id,
                # Metadata fields (keep for retrieval)
                "document_type": self.metadata.document_type,
                "category": self.metadata.category,
                "department": self.metadata.extra_metadata.get("department", "general"),
                "confidence_score": confidence_score,
                # Provenance fields
                "schema_id": "unity_docling_v1",
                "source_uri": self.metadata.file_path or "",
                "document_fingerprint": doc_fingerprint,
                "is_active": True,
            }

        # 1. Document-level record (root of the hierarchy)
        # Prefer a short, readable document id in the content_id path
        doc_content_id = self.document_id[:5]
        doc_title = (self.metadata.title or "").strip()
        doc_record = create_base_record(
            content_id=doc_content_id,
            content_type="document",
            title=doc_title,
            summary=self.summary,
            content_text=self.full_text,
            level=1,
            parent_id=None,
            confidence_score=self.metadata.confidence_score,
        )
        records.append(doc_record)

        # 2. Section-level records
        # Build helper maps to anchor images/tables under most likely section
        page_to_section_content_id: Dict[int, str] = {}
        section_content_ids: Dict[str, str] = {}
        # Map path-tuple to section_content_id for matching by path
        path_to_section_content_id: Dict[tuple, str] = {}
        for section in self.sections:
            # Use provided section_id (parser now uses short_id). Fallback to trimmed id.
            sec_short = section.section_id[:5]
            sec_key = f"{sec_short}"
            section_content_id = f"{doc_content_id}>{sec_key}"
            section_content_ids[section.section_id] = section_content_id
            section_record = create_base_record(
                content_id=section_content_id,
                content_type="section",
                title=(
                    f"{doc_title} > {section.title}" if doc_title else section.title
                ),
                summary=section.summary,
                content_text=section.content_text
                or "\n\n".join(p.text for p in section.paragraphs),
                level=2,
                parent_id=doc_content_id,
                confidence_score=section.metadata.get("confidence_score", 1.0),
            )
            records.append(section_record)

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
                    page_to_section_content_id[pn] = section_content_id
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
                    path_to_section_content_id[path] = section_content_id
            except Exception:
                pass

            # 3. Paragraph-level records
            for paragraph in section.paragraphs:
                para_short = paragraph.paragraph_id[:5]
                para_key = f"{para_short}"
                para_content_id = f"{section_content_id}>{para_key}"
                para_title = f"Paragraph {paragraph.paragraph_index + 1}"
                para_record = create_base_record(
                    content_id=para_content_id,
                    content_type="paragraph",
                    title=(
                        f"{doc_title} > {section.title} > {para_title}"
                        if doc_title
                        else f"{section.title} > {para_title}"
                    ),
                    summary=paragraph.summary,
                    content_text=paragraph.text,
                    level=3,
                    parent_id=section_content_id,
                    confidence_score=paragraph.metadata.get("confidence_score", 1.0),
                )
                records.append(para_record)

                # 4. Sentence-level records
                for sentence in paragraph.sentences:
                    sent_short = sentence.sentence_id[:5]
                    sent_key = f"{sent_short}"
                    sent_content_id = f"{para_content_id}>{sent_key}"
                    sent_title = f"Sentence {sentence.sentence_index + 1}"
                    sent_record = create_base_record(
                        content_id=sent_content_id,
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
                        level=4,
                        parent_id=para_content_id,
                        confidence_score=sentence.confidence_score,
                    )
                    records.append(sent_record)

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
                section_parent = None
                # a) path-based placement using parser-provided section_path
                try:
                    sec_path = getattr(img, "section_path", None)
                    if isinstance(sec_path, list) and sec_path:
                        tuple_path = tuple(
                            str(x).strip() for x in sec_path if str(x).strip()
                        )
                        # try exact, then progressively drop the deepest part
                        while tuple_path and section_parent is None:
                            if tuple_path in path_to_section_content_id:
                                section_parent = path_to_section_content_id[tuple_path]
                                break
                            tuple_path = tuple_path[:-1]
                except Exception:
                    pass

                # b) page-based placement
                if (
                    section_parent is None
                    and isinstance(page, int)
                    and page in page_to_section_content_id
                ):
                    section_parent = page_to_section_content_id[page]

                # c) best-effort: put under first section, else root doc
                if section_parent is None:
                    section_parent = (
                        section_content_ids.get(self.sections[0].section_id)
                        if self.sections
                        else doc_content_id
                    )

                # Content fields
                title = (
                    f"Image {idx + 1}{f' (page {page})' if page is not None else ''}"
                )
                image_content_id = f"{section_parent}>{short_id(5)}"

                # Resolve section title for this image
                try:
                    sec_title_for_img = next(
                        (
                            s.title
                            for s in self.sections
                            if section_content_ids.get(s.section_id) == section_parent
                        ),
                        None,
                    )
                except Exception:
                    sec_title_for_img = None

                img_record = create_base_record(
                    content_id=image_content_id,
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
                    level=3,
                    parent_id=section_parent,
                )
                # # include a few metadata points
                # img_record["page"] = page
                # img_record["bbox"] = (
                #     getattr(img, "bbox", None) if hasattr(img, "bbox") else None
                # )
                # img_record["element_type"] = (
                #     getattr(img, "element_type", None) if hasattr(img, "element_type") else None
                # )
                records.append(img_record)
            except Exception:
                continue

        # 6. Table-level records (children of sections)
        tables = getattr(self.metadata, "tables", []) or []
        for idx, tbl in enumerate(tables):
            try:
                page = getattr(tbl, "page", None) if hasattr(tbl, "page") else None
                section_parent = None
                # a) path-based placement using parser-provided section_path
                try:
                    sec_path = getattr(tbl, "section_path", None)
                    if isinstance(sec_path, list) and sec_path:
                        tuple_path = tuple(
                            str(x).strip() for x in sec_path if str(x).strip()
                        )
                        while tuple_path and section_parent is None:
                            if tuple_path in path_to_section_content_id:
                                section_parent = path_to_section_content_id[tuple_path]
                                break
                            tuple_path = tuple_path[:-1]
                except Exception:
                    pass

                # b) page-based placement
                if (
                    section_parent is None
                    and isinstance(page, int)
                    and page in page_to_section_content_id
                ):
                    section_parent = page_to_section_content_id[page]

                # c) fallback
                if section_parent is None:
                    section_parent = (
                        section_content_ids.get(self.sections[0].section_id)
                        if self.sections
                        else doc_content_id
                    )

                html = getattr(tbl, "html", None) if hasattr(tbl, "html") else None
                title = (
                    f"Table {idx + 1}{f' (page {page})' if page is not None else ''}"
                )
                content_text = (html or "").strip()
                table_content_id = f"{section_parent}>{short_id(5)}"

                # Resolve section title for this table
                try:
                    sec_title_for_tbl = next(
                        (
                            s.title
                            for s in self.sections
                            if section_content_ids.get(s.section_id) == section_parent
                        ),
                        None,
                    )
                except Exception:
                    sec_title_for_tbl = None

                tbl_record = create_base_record(
                    content_id=table_content_id,
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
                    level=3,
                    parent_id=section_parent,
                )
                # tbl_record["page"] = page
                # tbl_record["bbox"] = (
                #     getattr(tbl, "bbox", None) if hasattr(tbl, "bbox") else None
                # )
                # tbl_record["element_type"] = (
                #     getattr(tbl, "element_type", None) if hasattr(tbl, "element_type") else None
                # )
                records.append(tbl_record)
            except Exception:
                continue

        return records

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
