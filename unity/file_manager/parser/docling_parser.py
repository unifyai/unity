"""
Generic document parser with advanced features.

Supports multiple formats with features:
- Layout understanding and structure preservation
- Image extraction and storage
- Table structure recognition
- Hierarchical content organization
- Hybrid chunking with configurable text splitting
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Union

from .base import GenericParser
from .types.document import (
    Document,
    DocumentMetadata,
    DocumentMetadataExtraction,
    DocumentParagraph,
    DocumentSection,
    DocumentSentence,
)

# Check for optional dependencies
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import ConversionStatus
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

    # Placeholder classes for when Docling is not available
    class HybridChunker:
        pass

    class HuggingFaceTokenizer:
        pass


try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    RecursiveCharacterTextSplitter = None


class DoclingParser(GenericParser[Document]):
    """
    Advanced document parser with optional Docling backend support.
    Falls back to basic text parsing when advanced libraries are not available.

    This parser returns Document objects from the parse() method.
    """

    def __init__(
        self,
        *,
        max_chunk_size: int = 500,
        chunk_overlap: int = 200,
        sentence_chunk_size: int = 512,
        use_hybrid_chunking: bool = False,
        extract_images: bool = True,
        extract_tables: bool = True,
        use_llm_enrichment: bool = True,
        parser_name: str = "DoclingParser",
        parser_version: str = "1.0.0",
    ):
        """
        Initialize the document parser.

        Args:
            max_chunk_size: Maximum size for paragraph chunks (characters)
            chunk_overlap: Overlap between chunks (characters)
            sentence_chunk_size: Target size for individual sentences
            use_hybrid_chunking: Whether to use advanced hybrid chunking
            extract_images: Whether to extract and store images
            extract_tables: Whether to extract table data
            use_llm_enrichment: Whether to use LLM for metadata enrichment
            parser_name: Name of the parser
            parser_version: Version string
        """
        self.parser_name = parser_name
        self.parser_version = parser_version
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.sentence_chunk_size = sentence_chunk_size
        self.use_hybrid_chunking = use_hybrid_chunking and DOCLING_AVAILABLE
        self.extract_images = extract_images and DOCLING_AVAILABLE
        self.extract_tables = extract_tables and DOCLING_AVAILABLE
        self.use_llm_enrichment = use_llm_enrichment

        # Initialize converters if available
        self.converter = None
        if DOCLING_AVAILABLE:
            self.converter = DocumentConverter()

        # Initialize text splitters
        self.paragraph_splitter = None
        self.sentence_splitter = None
        if LANGCHAIN_AVAILABLE:
            self.paragraph_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            self.sentence_splitter = RecursiveCharacterTextSplitter(
                chunk_size=sentence_chunk_size,
                chunk_overlap=50,
                length_function=len,
                separators=[". ", "! ", "? ", "\n", " ", ""],
            )

        # Initialize hybrid chunker if requested
        self.hybrid_chunker = None
        if self.use_hybrid_chunking:
            self._init_hybrid_chunking()

        # Supported formats
        self.supported_formats = self._get_supported_formats()

    def _get_supported_formats(self) -> List[str]:
        """Get list of supported formats based on available backends."""
        if DOCLING_AVAILABLE:
            return [
                ".pdf",
                ".docx",
                ".txt",
            ]
        else:
            # Basic formats when only text parsing is available
            return [".txt"]

    def _get_mime_type(self, file_extension: str) -> str:
        """Convert file extension to MIME type."""
        mime_map = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".html": "text/html",
            ".htm": "text/html",
            ".json": "application/json",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".xml": "application/xml",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        return mime_map.get(file_extension.lower(), "application/octet-stream")

    def _init_hybrid_chunking(self) -> bool:
        """Initialize the hybrid chunker if available."""
        if not DOCLING_AVAILABLE or not LANGCHAIN_AVAILABLE:
            return False

        try:
            # Use a default embedding model tokenizer for chunking
            # Following Docling's best practices from documentation
            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2",
                ),
                max_tokens=self.max_chunk_size,
            )

            # Initialize hybrid chunker with best practice settings
            # Use smaller chunks to better preserve document structure
            self.hybrid_chunker = HybridChunker(
                tokenizer=tokenizer,
                max_tokens=min(
                    self.max_chunk_size,
                    300,
                ),  # Smaller chunks for better structure detection
                min_tokens=50,  # Avoid very small chunks
                merge_peers=True,  # Don't merge to preserve section boundaries
            )
            return True
        except Exception:
            self.hybrid_chunker = None
            return False

    def parse(self, file_path: Union[str, Path], /, **options: Any) -> Document:
        """
        Parse a document file into a structured Document object.

        Args:
            file_path: Path to the document file
            **options: Additional parser-specific options

        Returns:
            Document: Parsed document with hierarchical structure
        """
        print(f"Parsing document: {Path(file_path).name}")

        file_path = Path(file_path).expanduser().resolve()

        # Check if file exists
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(str(file_path))

        # Check if format is supported
        if file_path.suffix.lower() not in self.supported_formats:
            # Try basic text parsing as fallback
            return self._parse_as_text_document(file_path, **options)

        # For text files, always use basic parsing (Docling doesn't handle .txt well)
        if file_path.suffix.lower() in [".txt", ".log"]:
            document = self._parse_as_text_document(file_path, **options)
        # Use advanced parsing if available for other formats
        elif DOCLING_AVAILABLE and self.converter:
            try:
                document = self._parse_with_docling(file_path, **options)
            except Exception as e:
                # Fall back to basic parsing
                print(f"Advanced parsing failed, falling back to basic: {e}")
                document = self._parse_as_text_document(file_path, **options)
        else:
            document = self._parse_as_text_document(file_path, **options)

        print(
            f"Document parsed successfully: {len(document.sections)} sections, {document.get_total_paragraphs()} paragraphs",
        )

        # # Save parsed result if enabled
        # self._save_parsed_result_if_enabled(file_path, document)

        return document

    def _parse_with_docling(self, file_path: Path, **options: Any) -> Document:
        """Parse document using Docling with retry mechanism."""
        # First attempt with current settings
        try:
            return self._parse_document_docling(file_path, **options)
        except Exception as e:
            # If hybrid chunking is disabled and parsing failed, retry with hybrid chunking
            if not self.use_hybrid_chunking:
                try:
                    # Temporarily enable hybrid chunking
                    original_setting = self.use_hybrid_chunking
                    self.use_hybrid_chunking = True

                    # Re-initialize hybrid chunker if not already done
                    if self.hybrid_chunker is None:
                        if not self._init_hybrid_chunking():
                            self.use_hybrid_chunking = original_setting
                            raise e

                    # Retry parsing
                    result = self._parse_document_docling(file_path, **options)

                    # Restore original setting
                    self.use_hybrid_chunking = original_setting
                    return result

                except Exception:
                    # Restore original setting
                    self.use_hybrid_chunking = original_setting
                    raise

            else:
                raise

    def _parse_document_docling(self, file_path: Path, **options: Any) -> Document:
        """Internal method to parse a document using Docling."""
        start_time = time.time()

        # Convert document using Docling
        result = self.converter.convert(source=str(file_path))

        if result.status != ConversionStatus.SUCCESS:
            raise ValueError(f"Document conversion failed with status: {result.status}")

        docling_doc = result.document

        # Create base metadata
        metadata = self._create_base_metadata(file_path)
        metadata.processing_time = time.time() - start_time

        # Extract document statistics
        full_text = docling_doc.export_to_text()
        metadata.total_characters = len(full_text)
        metadata.total_words = len(full_text.split())

        # Extract page count from Docling document
        try:
            if hasattr(docling_doc, "pages") and docling_doc.pages:
                metadata.total_pages = len(docling_doc.pages)
            elif hasattr(result, "pages"):
                metadata.total_pages = len(result.pages)
        except Exception:
            pass

        # Create document
        document_id = str(hash(str(file_path)) % (10**10))
        document = Document(
            document_id=document_id,
            metadata=metadata,
            full_text=full_text,
            processing_status="processing",
        )

        # Extract content structure
        self._extract_document_structure(docling_doc, document)

        # Extract images if requested
        if self.extract_images:
            self._extract_images(docling_doc, document)

        # Extract tables if requested
        if self.extract_tables:
            self._extract_tables(docling_doc, document)

        # Generate hierarchical summaries
        if self.use_llm_enrichment:
            print("Generating summaries...")
            self._generate_summaries(document)

        # Extract enhanced metadata using LLM if enabled
        if self.use_llm_enrichment:
            self._extract_enhanced_metadata(document)

        # Update statistics
        self._update_statistics(document)

        document.processing_status = "completed"
        return document

    def _parse_as_text_document(self, file_path: Path, **options: Any) -> Document:
        """Parse as basic text document."""
        start_time = time.time()

        # Read file content
        try:
            if file_path.suffix.lower() == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.dumps(json.load(f), indent=2)
            else:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            raise ValueError(f"Failed to read file: {e}")

        # Create metadata
        metadata = self._create_base_metadata(file_path)
        metadata.processing_time = time.time() - start_time
        metadata.total_characters = len(content)
        metadata.total_words = len(content.split())

        # Create document
        document_id = str(hash(str(file_path)) % (10**10))
        document = Document(
            document_id=document_id,
            metadata=metadata,
            full_text=content,
            processing_status="processing",
        )

        # Extract basic structure
        self._extract_basic_structure(content, document)

        # Update statistics
        self._update_statistics(document)

        document.processing_status = "completed"
        return document

    def _create_base_metadata(
        self,
        file_path: Path,
        clean_temp_path: bool = True,
    ) -> DocumentMetadata:
        """Create base metadata from file information."""
        from datetime import datetime

        stat_info = file_path.stat()

        # Clean up temp directory from path if requested
        path_str = str(file_path)
        if clean_temp_path and "/tmp/" in path_str:
            # Extract just the meaningful part after /tmp/
            parts = path_str.split("/tmp/")
            if len(parts) > 1:
                # Get everything after the tmp directory
                after_tmp = parts[1]
                # Skip the temp folder name (first part) to get original filename
                subparts = after_tmp.split("/", 1)
                if len(subparts) > 1:
                    path_str = subparts[1]
                else:
                    # If no subpath, just use the filename
                    path_str = file_path.name

        return DocumentMetadata(
            title=file_path.stem,
            file_path=path_str,
            file_name=file_path.name,
            file_size=stat_info.st_size,
            file_type=self._get_mime_type(file_path.suffix.lower()),
            created_at=datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
            modified_at=datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            processed_at=datetime.now().isoformat(),
            parser_name=self.parser_name,
            parser_version=self.parser_version,
        )

    def _extract_document_structure(self, docling_doc, document: Document):
        """Extract hierarchical document structure with waterfall fallback."""
        extraction_successful = False

        # Waterfall fallback structure
        # 1. Try hybrid chunking if enabled
        if self.use_hybrid_chunking and self.hybrid_chunker:
            try:
                self._extract_with_hybrid_chunking(docling_doc, document)
                extraction_successful = True
            except Exception as e:
                pass

        # 2. Fall back to native Docling structure
        if (
            not extraction_successful
            and hasattr(docling_doc, "body")
            and docling_doc.body
        ):
            try:
                self._extract_native_structure(docling_doc, document)
                extraction_successful = True
            except Exception as e:
                pass

        # 3. Fall back to text splitting
        if not extraction_successful:
            try:
                self._extract_with_text_splitting(docling_doc, document)
                extraction_successful = True
            except Exception as e:
                pass

        # 4. Final fallback to basic structure
        if not extraction_successful:
            try:
                full_text = docling_doc.export_to_text()
                self._extract_basic_structure(full_text, document)
            except Exception as e:
                pass
                # Ensure document has at least minimal structure
                if not document.sections:
                    section = DocumentSection(
                        section_id=f"{document.document_id}_section_1",
                        title="Document Content",
                        content=document.content or "",
                        paragraphs=[],
                    )
                    document.sections.append(section)

    def _extract_native_structure(self, docling_doc, document: Document):
        """Extract structure using Docling's native document model with better heading detection."""
        section_id = 1
        paragraph_id = 1
        sentence_id = 1

        # Track current section for proper hierarchy
        current_section = None
        section_stack = []  # Stack of (level, section) for hierarchical tracking

        # Buffer for accumulating text items into paragraphs
        paragraph_buffer = []
        paragraph_metadata = {}

        def flush_paragraph_buffer():
            """Helper to create paragraph from buffered text items."""
            nonlocal paragraph_id, sentence_id
            if paragraph_buffer and current_section:
                # Combine buffered text
                combined_text = " ".join(paragraph_buffer)

                # Create paragraph
                paragraph = DocumentParagraph(
                    text=combined_text,
                    paragraph_id=str(paragraph_id),
                    section_id=current_section.section_id,
                    document_id=document.document_id,
                    paragraph_index=len(current_section.paragraphs),
                    metadata=paragraph_metadata.copy(),
                )

                # Split into sentences
                sentences = self._split_into_sentences(
                    combined_text,
                    paragraph_id,
                    current_section.section_id,
                    document.document_id,
                    sentence_id,
                )
                paragraph.sentences = sentences
                sentence_id += len(sentences)

                current_section.paragraphs.append(paragraph)
                paragraph_id += 1

                # Clear buffer
                paragraph_buffer.clear()
                paragraph_metadata.clear()

        # Use Docling's iterate_items method to traverse in reading order
        try:
            # Iterate through all items including groups
            # The 'level' parameter gives us hierarchy information!
            for item, level in docling_doc.iterate_items(
                with_groups=True,
                traverse_pictures=False,
            ):
                # Skip empty items
                if hasattr(item, "text") and not item.text.strip():
                    continue

                # Handle different item types based on Docling's type system
                item_type = type(item).__name__

                # Check if this is a heading based on label or type
                is_heading = False

                # Method 1: Check item type
                if item_type in ["TitleItem", "SectionHeaderItem", "HeadingItem"]:
                    is_heading = True

                # Method 2: Check label attribute (Docling's semantic labeling)
                elif hasattr(item, "label") and item.label:
                    label_lower = str(item.label).lower()
                    if any(
                        heading_indicator in label_lower
                        for heading_indicator in [
                            "title",
                            "heading",
                            "header",
                            "section",
                            "chapter",
                            "part",
                            "h1",
                            "h2",
                            "h3",
                            "h4",
                            "h5",
                            "h6",
                        ]
                    ):
                        is_heading = True

                # Method 3: Check text pattern if not identified as heading yet
                elif hasattr(item, "text") and item.text:
                    text = item.text.strip()
                    if self._is_likely_header(text, None):
                        is_heading = True

                if is_heading:
                    # Flush any pending paragraph before starting new section
                    flush_paragraph_buffer()

                    # Extract text and level
                    title_text = (
                        item.text.strip() if hasattr(item, "text") else "Untitled"
                    )
                    # Use the iterate_items level parameter for hierarchy
                    header_level = (
                        level if level is not None else getattr(item, "level", 1)
                    )

                    # Pop sections from stack for proper hierarchy
                    while section_stack and section_stack[-1][0] >= header_level:
                        section_stack.pop()

                    # Create new section
                    current_section = DocumentSection(
                        title=title_text,
                        section_id=str(section_id),
                        document_id=document.document_id,
                        section_index=section_id - 1,
                        metadata={
                            "level": header_level,
                            "docling_type": item_type,
                            "docling_label": (
                                str(item.label) if hasattr(item, "label") else None
                            ),
                        },
                    )
                    document.sections.append(current_section)
                    section_stack.append((header_level, current_section))
                    section_id += 1

                # Handle all other text-containing items (TextItem, etc.)
                elif hasattr(item, "text") and item.text.strip():
                    # Ensure we have a section
                    if current_section is None:
                        current_section = DocumentSection(
                            title="Document Content",
                            section_id=str(section_id),
                            document_id=document.document_id,
                            section_index=0,
                        )
                        document.sections.append(current_section)
                        section_stack.append((0, current_section))
                        section_id += 1

                    text = item.text.strip()

                    # Determine if this text should start a new paragraph
                    # (e.g., after significant whitespace, different formatting, etc.)
                    should_start_new_paragraph = False

                    # Check if this seems like a paragraph boundary
                    if paragraph_buffer:
                        last_text = paragraph_buffer[-1]
                        # New paragraph if:
                        # - Previous text ended with sentence ending
                        # - Current text starts with capital letter
                        # - There's a style/formatting change
                        if (
                            last_text.rstrip().endswith((".", "!", "?", ":"))
                            and text
                            and text[0].isupper()
                        ):
                            should_start_new_paragraph = True
                        # Also check for list items or special formatting
                        elif text.startswith(
                            (
                                "•",
                                "▪",
                                "◦",
                                "-",
                                "*",
                                "1.",
                                "2.",
                                "3.",
                                "4.",
                                "5.",
                                "6.",
                                "7.",
                                "8.",
                                "9.",
                                "a.",
                                "b.",
                                "c.",
                                "d.",
                                "e.",
                            ),
                        ):
                            should_start_new_paragraph = True

                    if should_start_new_paragraph:
                        flush_paragraph_buffer()

                    # Add to buffer
                    paragraph_buffer.append(text)

                    # Update metadata
                    if not paragraph_metadata:
                        paragraph_metadata["docling_type"] = item_type
                        paragraph_metadata["docling_label"] = (
                            str(item.label) if hasattr(item, "label") else None
                        )

                        # Extract provenance for first item
                        if hasattr(item, "prov") and item.prov:
                            for prov_item in item.prov:
                                if hasattr(prov_item, "page_no"):
                                    paragraph_metadata["page_no"] = prov_item.page_no
                                if hasattr(prov_item, "bbox"):
                                    # Store bbox as dict
                                    if hasattr(prov_item.bbox, "model_dump"):
                                        paragraph_metadata["bbox"] = (
                                            prov_item.bbox.model_dump()
                                        )
                                    else:
                                        paragraph_metadata["bbox"] = {
                                            "l": getattr(prov_item.bbox, "l", 0),
                                            "t": getattr(prov_item.bbox, "t", 0),
                                            "r": getattr(prov_item.bbox, "r", 0),
                                            "b": getattr(prov_item.bbox, "b", 0),
                                        }
                                break

                # Handle ListItem
                elif item_type == "ListItem" and hasattr(item, "text"):
                    # Flush any pending paragraph before list item
                    flush_paragraph_buffer()

                    if current_section is None:
                        current_section = DocumentSection(
                            title="Document Content",
                            section_id=str(section_id),
                            document_id=document.document_id,
                            section_index=0,
                        )
                        document.sections.append(current_section)
                        section_stack.append((0, current_section))
                        section_id += 1

                    # Create paragraph with list metadata
                    paragraph = DocumentParagraph(
                        text=item.text.strip(),
                        paragraph_id=str(paragraph_id),
                        section_id=current_section.section_id,
                        document_id=document.document_id,
                        paragraph_index=len(current_section.paragraphs),
                        metadata={
                            "type": "list_item",
                            "enumerated": getattr(item, "enumerated", False),
                            "marker": getattr(item, "marker", ""),
                            "docling_type": item_type,
                            "docling_label": (
                                str(item.label) if hasattr(item, "label") else None
                            ),
                        },
                    )

                    # Add provenance if available
                    if hasattr(item, "prov") and item.prov:
                        for prov_item in item.prov:
                            if hasattr(prov_item, "page_no"):
                                paragraph.metadata["page_no"] = prov_item.page_no
                            break

                    # Simple sentence for list items
                    paragraph.sentences = [
                        DocumentSentence(
                            text=item.text.strip(),
                            sentence_id=str(sentence_id),
                            paragraph_id=str(paragraph_id),
                            section_id=current_section.section_id,
                            document_id=document.document_id,
                            sentence_index=0,
                        ),
                    ]
                    sentence_id += 1

                    current_section.paragraphs.append(paragraph)
                    paragraph_id += 1

                # Handle CodeItem and FormulaItem
                elif item_type in ["CodeItem", "FormulaItem"] and hasattr(item, "text"):
                    if current_section is None:
                        current_section = DocumentSection(
                            title="Document Content",
                            section_id=str(section_id),
                            document_id=document.document_id,
                            section_index=0,
                        )
                        document.sections.append(current_section)
                        section_stack.append((0, current_section))
                        section_id += 1

                    # Create specialized paragraph
                    paragraph = DocumentParagraph(
                        text=item.text.strip(),
                        paragraph_id=str(paragraph_id),
                        section_id=current_section.section_id,
                        document_id=document.document_id,
                        paragraph_index=len(current_section.paragraphs),
                        metadata={
                            "type": "code" if item_type == "CodeItem" else "formula",
                            "code_language": (
                                getattr(item, "code_language", None)
                                if item_type == "CodeItem"
                                else None
                            ),
                            "docling_type": item_type,
                            "docling_label": (
                                str(item.label) if hasattr(item, "label") else None
                            ),
                        },
                    )

                    # Don't split code/formula into sentences
                    paragraph.sentences = [
                        DocumentSentence(
                            text=item.text.strip(),
                            sentence_id=str(sentence_id),
                            paragraph_id=str(paragraph_id),
                            section_id=current_section.section_id,
                            document_id=document.document_id,
                            sentence_index=0,
                        ),
                    ]
                    sentence_id += 1

                    current_section.paragraphs.append(paragraph)
                    paragraph_id += 1

            # Flush any remaining buffered text at the end
            flush_paragraph_buffer()

        except AttributeError:
            # Fallback if iterate_items is not available
            # Try alternative approach using texts, tables, etc. collections
            self._extract_from_collections(docling_doc, document)

    def _extract_from_collections(self, docling_doc, document: Document):
        """Fallback extraction using Docling's text collections."""
        section_id = 1
        paragraph_id = 1
        sentence_id = 1
        current_section = None

        # Process texts collection if available
        if hasattr(docling_doc, "texts"):
            for text_item in docling_doc.texts:
                if not hasattr(text_item, "text") or not text_item.text.strip():
                    continue

                # Check item type by class name
                item_type = type(text_item).__name__

                if item_type in ["TitleItem", "SectionHeaderItem"]:
                    # Create new section
                    current_section = DocumentSection(
                        title=text_item.text.strip(),
                        section_id=str(section_id),
                        document_id=document.document_id,
                        section_index=section_id - 1,
                        metadata={
                            "level": (
                                getattr(text_item, "level", 1)
                                if item_type == "SectionHeaderItem"
                                else 0
                            ),
                            "docling_type": item_type,
                        },
                    )
                    document.sections.append(current_section)
                    section_id += 1
                else:
                    # Regular text content
                    if current_section is None:
                        current_section = DocumentSection(
                            title="Document Content",
                            section_id=str(section_id),
                            document_id=document.document_id,
                            section_index=0,
                        )
                        document.sections.append(current_section)
                        section_id += 1

                    # Create paragraph
                    paragraph = DocumentParagraph(
                        text=text_item.text.strip(),
                        paragraph_id=str(paragraph_id),
                        section_id=current_section.section_id,
                        document_id=document.document_id,
                        paragraph_index=len(current_section.paragraphs),
                        metadata={"docling_type": item_type},
                    )

                    # Add provenance
                    if hasattr(text_item, "prov") and text_item.prov:
                        for prov in text_item.prov:
                            if hasattr(prov, "page_no"):
                                paragraph.metadata["page_no"] = prov.page_no
                            break

                    # Split into sentences
                    sentences = self._split_into_sentences(
                        text_item.text.strip(),
                        paragraph_id,
                        current_section.section_id,
                        document.document_id,
                        sentence_id,
                    )
                    paragraph.sentences = sentences
                    sentence_id += len(sentences)

                    current_section.paragraphs.append(paragraph)
                    paragraph_id += 1

    def _extract_with_hybrid_chunking(self, docling_doc, document: Document):
        """Extract structure using Docling's hybrid chunker with enhanced context awareness."""
        # Get all chunks from the hybrid chunker
        chunks = list(self.hybrid_chunker.chunk(docling_doc))

        # Store chunks for potential enrichment in other methods
        self._last_chunks = chunks
        self._chunk_text_map = {}  # Map chunk text to enriched context

        # Build a mapping of chunks to their hierarchical context
        chunk_hierarchy = self._build_chunk_hierarchy(chunks)

        section_id = 1
        paragraph_id = 1
        sentence_id = 1

        # Track sections by their hierarchical path
        section_stack = []  # Stack of (level, section) tuples
        current_section = None

        # Track the last seen headings hierarchy
        last_headings = []

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.text.strip()
            if not chunk_text:
                continue

            # Get hierarchical context from chunk
            chunk_path = chunk_hierarchy.get(i, {})
            is_heading = chunk_path.get("is_heading", False)
            heading_level = chunk_path.get("level", 0)

            # Check if headings changed (indicates new section)
            current_headings = []
            if (
                hasattr(chunk, "meta")
                and chunk.meta
                and hasattr(chunk.meta, "headings")
            ):
                current_headings = chunk.meta.headings or []

            # If headings changed from last chunk, we might be in a new section
            if current_headings != last_headings and current_headings:
                # Create sections for any new headings
                for idx, heading in enumerate(current_headings):
                    if idx >= len(last_headings) or heading != last_headings[idx]:
                        # This is a new heading at this level

                        # Pop sections from stack for proper hierarchy
                        while section_stack and section_stack[-1][0] >= idx:
                            section_stack.pop()

                        # Create section for this heading
                        new_section = DocumentSection(
                            title=heading,
                            section_id=str(section_id),
                            document_id=document.document_id,
                            section_index=section_id - 1,
                            metadata={
                                "level": idx,
                                "from_chunk_headings": True,
                            },
                        )
                        document.sections.append(new_section)
                        section_stack.append((idx, new_section))
                        current_section = new_section
                        section_id += 1

            last_headings = current_headings

            # More aggressive heading detection
            # First check metadata
            if hasattr(chunk, "meta") and chunk.meta:
                # Check for heading indicators in metadata
                # DocMeta has doc_items which contain labels
                if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for doc_item in chunk.meta.doc_items:
                        if hasattr(doc_item, "label") and doc_item.label:
                            label_lower = doc_item.label.lower()
                            if label_lower in [
                                "title",
                                "heading",
                                "section_header",
                                "h1",
                                "h2",
                                "h3",
                                "header",
                            ]:
                                is_heading = True
                                break
                    # Extract level from label if possible
                    for doc_item in chunk.meta.doc_items:
                        if hasattr(doc_item, "label") and doc_item.label:
                            label = doc_item.label.lower()
                            if label.startswith("h") and label[1:].isdigit():
                                heading_level = int(label[1:])

            # Always check text patterns - don't skip if metadata didn't identify it
            if self._is_likely_header(chunk_text, chunk):
                is_heading = True
                if heading_level == 0:  # Only set if not already set from metadata
                    heading_level = self._estimate_heading_level(chunk_text)

            if is_heading:
                # Pop sections from stack until we find the right level
                while section_stack and section_stack[-1][0] >= heading_level:
                    section_stack.pop()

                # Create new section
                current_section = DocumentSection(
                    title=chunk_text,
                    section_id=str(section_id),
                    document_id=document.document_id,
                    section_index=section_id - 1,
                    metadata={
                        "level": heading_level,
                        "path": [s[1].title for s in section_stack] + [chunk_text],
                    },
                )
                document.sections.append(current_section)
                section_stack.append((heading_level, current_section))
                section_id += 1

            else:
                # This is content
                if current_section is None:
                    # Create default section if none exists
                    current_section = DocumentSection(
                        title="Document Content",
                        section_id=str(section_id),
                        document_id=document.document_id,
                        section_index=0,
                    )
                    document.sections.append(current_section)
                    section_stack.append((0, current_section))
                    section_id += 1

                # Validate chunk token count
                if not self.validate_chunk_tokens(chunk):
                    print(
                        f"Warning: Chunk {i} exceeds token limit, may need further splitting",
                    )

                # Store enriched context for potential later use
                enriched_text = self.enrich_chunk_context(chunk)
                self._chunk_text_map[chunk_text] = enriched_text

                # Create paragraph from chunk with enriched metadata
                paragraph = DocumentParagraph(
                    text=chunk_text,
                    paragraph_id=str(paragraph_id),
                    section_id=current_section.section_id,
                    document_id=document.document_id,
                    paragraph_index=len(current_section.paragraphs),
                )

                # Get comprehensive chunk metadata using helper method
                chunk_metadata = self.get_chunk_metadata(chunk)
                paragraph.metadata.update(
                    {
                        "chunk_index": i,
                        "chunk_valid_tokens": self.validate_chunk_tokens(chunk),
                        "enriched_context": enriched_text,  # Store enriched text for embeddings
                        "enriched_length": len(enriched_text),
                        **chunk_metadata,  # Include all metadata from helper
                    },
                )

                # Add chunk metadata for provenance tracking
                if hasattr(chunk, "meta") and chunk.meta:
                    # Extract provenance from doc_items
                    if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                        pages = set()
                        bboxes = []

                        for doc_item in chunk.meta.doc_items:
                            if hasattr(doc_item, "prov"):
                                for prov in doc_item.prov:
                                    if hasattr(prov, "page_no"):
                                        pages.add(prov.page_no)
                                    if hasattr(prov, "bbox"):
                                        bbox_dict = {
                                            "l": getattr(prov.bbox, "l", 0),
                                            "t": getattr(prov.bbox, "t", 0),
                                            "r": getattr(prov.bbox, "r", 0),
                                            "b": getattr(prov.bbox, "b", 0),
                                        }
                                        bboxes.append(bbox_dict)

                        if pages:
                            paragraph.metadata["pages"] = sorted(list(pages))
                        if bboxes:
                            paragraph.metadata["bboxes"] = bboxes

                # Split paragraph into sentences with context awareness
                sentences = self._split_into_sentences_contextual(
                    chunk_text,
                    paragraph_id,
                    current_section.section_id,
                    document.document_id,
                    sentence_id,
                    chunk,
                )
                paragraph.sentences = sentences
                sentence_id += len(sentences)

                current_section.paragraphs.append(paragraph)
                paragraph_id += 1

    def _build_chunk_hierarchy(self, chunks) -> Dict[int, Dict]:
        """Build hierarchical context for chunks using HybridChunker metadata."""
        hierarchy = {}

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.text.strip()
            context = {
                "is_heading": False,
                "level": 0,
                "parent_heading": None,
                "doc_items": [],
                "headings": [],
            }

            # First check: Text-based heuristics for headings
            # These are often more reliable than metadata for PDFs
            import re

            if chunk_text and len(chunk_text) < 200:  # Headings are usually short
                # Pattern 1: Numbered sections (e.g., "1. Introduction", "2.1 Background")
                if re.match(r"^[\d\.]+\s+[A-Z]", chunk_text):
                    context["is_heading"] = True
                    context["level"] = chunk_text.count(".") + 1

                # Pattern 2: ALL CAPS
                elif chunk_text.isupper() and len(chunk_text.split()) < 10:
                    context["is_heading"] = True
                    context["level"] = 1

                # Pattern 3: Title Case without sentence ending
                elif (
                    re.match(r"^[A-Z][^.!?]*$", chunk_text)
                    and len(chunk_text.split()) < 15
                    and not chunk_text.endswith((":", ","))
                ):
                    context["is_heading"] = True
                    context["level"] = 2

                # Pattern 4: Common section markers
                elif re.match(
                    r"^(Chapter|Section|Part|Appendix)\s+[\dIVXLC]+",
                    chunk_text,
                    re.I,
                ):
                    context["is_heading"] = True
                    context["level"] = 1

            # Extract metadata from DocChunk structure (secondary check)
            if hasattr(chunk, "meta") and chunk.meta:
                # Get headings from meta (hierarchical path)
                if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    context["headings"] = chunk.meta.headings
                    # If metadata says it's under headings, it might be a heading itself
                    if not context["is_heading"] and chunk.meta.headings:
                        # Check if this chunk's text matches the last heading
                        if chunk_text in chunk.meta.headings[-1]:
                            context["is_heading"] = True
                            context["level"] = len(chunk.meta.headings)

                # Get document items from meta
                if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    context["doc_items"] = chunk.meta.doc_items

                    # Analyze doc items to determine if this is a heading
                    for doc_item in chunk.meta.doc_items:
                        if hasattr(doc_item, "label"):
                            label = str(doc_item.label).lower()
                            if label in [
                                "title",
                                "heading",
                                "section_header",
                            ] or label.startswith("h"):
                                context["is_heading"] = True
                                if hasattr(doc_item, "level"):
                                    context["level"] = doc_item.level
                                elif (
                                    label.startswith("h")
                                    and len(label) > 1
                                    and label[1].isdigit()
                                ):
                                    context["level"] = int(label[1])
                                elif label == "title":
                                    context["level"] = 0
                                else:
                                    context["level"] = 1
                                break

            hierarchy[i] = context

        return hierarchy

    def _is_likely_header(self, text: str, chunk) -> bool:
        """Enhanced header detection using multiple signals - more aggressive."""
        text = text.strip()
        if not text:
            return False

        # Check text characteristics
        if len(text) > 200:  # Too long for header
            return False

        import re

        text_lower = text.lower()

        # Common section keywords at start
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
        if any(text_lower.startswith(keyword) for keyword in section_keywords):
            return True

        # Check formatting patterns
        if text.isupper() and len(text.split()) < 15:  # All caps
            return True

        # Short text without ending punctuation (common for headings)
        if len(text) < 60 and not text.endswith(
            (".", ",", ";", ":", "!", "?", '"', "'"),
        ):
            if text[0].isupper():  # Starts with capital
                return True

        # Title case check
        if text.istitle():
            return True

        # Markdown headers
        if text.startswith(("#", "##", "###")):
            return True

        # Various numbering patterns
        if re.match(r"^[\d\.]+\s+", text):  # 1. 1.1 1.1.1 etc
            return True
        if re.match(r"^[A-Z]\.\s+", text):  # A. B. C.
            return True
        if re.match(r"^\([a-zA-Z0-9]+\)\s+", text):  # (a) (1) (A)
            return True
        if re.match(r"^[IVXLCDM]+\.?\s+", text):  # Roman numerals
            return True
        if re.match(r"^(Chapter|Section|Part)\s+[\dIVXLCDM]+", text, re.I):
            return True

        # Check chunk metadata for additional hints
        if hasattr(chunk, "meta") and chunk.meta:
            # Check doc_items for font/style hints
            if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                for doc_item in chunk.meta.doc_items:
                    # Check if any doc_item indicates heading characteristics
                    if hasattr(doc_item, "label") and doc_item.label:
                        # Labels like 'title', 'heading' indicate headers
                        if doc_item.label.lower() in [
                            "title",
                            "heading",
                            "section_header",
                        ]:
                            return True

        return False

    def _estimate_heading_level(self, text: str) -> int:
        """Estimate heading level from text patterns."""
        # Markdown-style headers
        if text.startswith("###"):
            return 3
        elif text.startswith("##"):
            return 2
        elif text.startswith("#"):
            return 1

        # Numbered sections
        import re

        match = re.match(r"^(\d+)\.", text)
        if match:
            # Main sections (1., 2., etc.) are level 1
            return 1

        # Subsections (1.1, 2.3, etc.)
        match = re.match(r"^\d+\.\d+", text)
        if match:
            return 2

        # Default level for other headers
        return 2

    def _split_into_sentences_contextual(
        self,
        text: str,
        paragraph_id: Union[int, str],
        section_id: Union[int, str],
        document_id: Union[int, str],
        start_sentence_id: int,
        chunk,
    ) -> List[DocumentSentence]:
        """Split text into sentences with awareness of chunk context."""
        # Use regular splitting but add chunk metadata to sentences
        sentences = self._split_into_sentences(
            text,
            paragraph_id,
            section_id,
            document_id,
            start_sentence_id,
        )

        # Enrich sentences with chunk context
        if hasattr(chunk, "meta") and chunk.meta:
            for sentence in sentences:
                sentence.metadata = sentence.metadata or {}
                # Get label from first doc_item if available
                chunk_label = "text"
                if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    first_item = chunk.meta.doc_items[0]
                    if hasattr(first_item, "label") and first_item.label:
                        chunk_label = first_item.label
                sentence.metadata["chunk_type"] = chunk_label

        return sentences

    def enrich_chunk_context(self, chunk) -> str:
        """Enrich chunk with hierarchical context using HybridChunker's contextualize method."""
        if self.hybrid_chunker and hasattr(self.hybrid_chunker, "contextualize"):
            try:
                # Use HybridChunker's built-in contextualization
                # This adds headings and metadata to provide full context
                return self.hybrid_chunker.contextualize(chunk)
            except Exception as e:

                # Fallback to chunk text
                return chunk.text if hasattr(chunk, "text") else str(chunk)
        return chunk.text if hasattr(chunk, "text") else str(chunk)

    def get_chunk_metadata(self, chunk) -> Dict[str, Any]:
        """Extract comprehensive metadata from a HybridChunker chunk."""
        metadata = {}

        if hasattr(chunk, "meta") and chunk.meta:
            # Extract headings hierarchy
            if hasattr(chunk.meta, "headings"):
                metadata["headings"] = chunk.meta.headings

            # Extract document items info
            if hasattr(chunk.meta, "doc_items"):
                metadata["doc_items_count"] = len(chunk.meta.doc_items)

                # Collect item types
                item_types = []
                for item in chunk.meta.doc_items:
                    if hasattr(item, "label"):
                        item_types.append(str(item.label))
                metadata["item_types"] = item_types

            # Extract origin info
            if hasattr(chunk.meta, "origin"):
                metadata["origin"] = chunk.meta.origin

        return metadata

    def validate_chunk_tokens(self, chunk) -> bool:
        """Validate that chunk is within token limits using HybridChunker's tokenizer."""
        if self.hybrid_chunker and hasattr(self.hybrid_chunker, "tokenizer"):
            try:
                # Get the contextualized text (with headings and metadata)
                contextualized_text = self.enrich_chunk_context(chunk)
                # Count tokens using the chunker's tokenizer
                token_count = self.hybrid_chunker.tokenizer.count_tokens(
                    contextualized_text,
                )
                # Check against max_tokens
                return token_count <= self.hybrid_chunker.max_tokens
            except Exception:
                # If validation fails, assume it's valid
                return True
        return True

    def get_enriched_text(self, text: str) -> str:
        """Get enriched text with hierarchical context if available."""
        # Check if we have a mapping from chunk extraction
        if hasattr(self, "_chunk_text_map") and text in self._chunk_text_map:
            return self._chunk_text_map[text]
        # Otherwise return original text
        return text

    def _extract_with_text_splitting(self, docling_doc, document: Document):
        """Extract structure using text splitters."""
        # Export to markdown to preserve some structure
        markdown_text = docling_doc.export_to_markdown()

        # Split into sections based on headers
        sections = self._split_by_headers(markdown_text)

        section_id = 1
        paragraph_id = 1
        sentence_id = 1

        for section_title, section_content in sections:
            section = DocumentSection(
                title=section_title or f"Section {section_id}",
                section_id=str(section_id),
                document_id=document.document_id,
                section_index=section_id - 1,
            )

            # Split section into paragraphs
            if self.paragraph_splitter:
                paragraph_chunks = self.paragraph_splitter.split_text(section_content)
            else:
                # Simple paragraph splitting
                paragraph_chunks = section_content.split("\n\n")

            for chunk_text in paragraph_chunks:
                if not chunk_text.strip():
                    continue

                paragraph = DocumentParagraph(
                    text=chunk_text.strip(),
                    paragraph_id=str(paragraph_id),
                    section_id=section.section_id,
                    document_id=document.document_id,
                    paragraph_index=len(section.paragraphs),
                )

                # Split paragraph into sentences
                sentences = self._split_into_sentences(
                    chunk_text.strip(),
                    paragraph_id,
                    section.section_id,
                    document.document_id,
                    sentence_id,
                )
                paragraph.sentences = sentences
                sentence_id += len(sentences)

                section.paragraphs.append(paragraph)
                paragraph_id += 1

            document.sections.append(section)
            section_id += 1

    def _extract_basic_structure(self, text: str, document: Document):
        """Enhanced basic structure extraction with intelligent paragraph detection."""
        # Try to detect basic structure even in plain text
        lines = text.split("\n")
        sections = []
        current_section_lines = []
        current_title = None

        section_id = 1
        paragraph_id = 1
        sentence_id = 1

        # Patterns that might indicate section headers
        import re

        header_patterns = [
            re.compile(r"^#{1,6}\s+(.+)$"),  # Markdown headers
            re.compile(r"^([A-Z][A-Z\s]+)$"),  # ALL CAPS HEADERS
            re.compile(r"^(\d+\.?\s+[A-Z].+)$"),  # Numbered sections
            re.compile(r"^([A-Z][^.!?]*):?\s*$"),  # Title case followed by colon
        ]

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Check if this line might be a header
            is_header = False
            header_text = None

            for pattern in header_patterns:
                match = pattern.match(line_stripped)
                if match:
                    # Additional checks to avoid false positives
                    potential_header = (
                        match.group(1) if match.lastindex else line_stripped
                    )
                    if len(potential_header) < 100 and not potential_header.endswith(
                        ".",
                    ):
                        is_header = True
                        header_text = potential_header.strip("#").strip()
                        break

            # Also check for lines that are followed by underlines
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and (set(next_line) <= {"=", "-"} and len(next_line) >= 3):
                    is_header = True
                    header_text = line_stripped

            if is_header and header_text:
                # Save previous section if exists
                if current_section_lines or current_title:
                    sections.append((current_title, "\n".join(current_section_lines)))

                current_title = header_text
                current_section_lines = []
            else:
                current_section_lines.append(line)

        # Add final section
        if current_section_lines or current_title:
            sections.append((current_title, "\n".join(current_section_lines)))

        # If no sections detected, treat as single section
        if not sections:
            sections = [("Document Content", text)]

        # Process sections
        for section_title, section_content in sections:
            if not section_content.strip() and not section_title:
                continue

            section = DocumentSection(
                title=section_title or f"Section {section_id}",
                section_id=str(section_id),
                document_id=document.document_id,
                section_index=section_id - 1,
            )

            # Split section content into paragraphs
            if self.paragraph_splitter and section_content.strip():
                paragraph_chunks = self.paragraph_splitter.split_text(section_content)
            else:
                # Enhanced paragraph splitting
                paragraph_chunks = self._split_into_paragraphs(section_content)

            for chunk_text in paragraph_chunks:
                if not chunk_text.strip():
                    continue

                paragraph = DocumentParagraph(
                    text=chunk_text.strip(),
                    paragraph_id=str(paragraph_id),
                    section_id=section.section_id,
                    document_id=document.document_id,
                    paragraph_index=len(section.paragraphs),
                )

                # Split paragraph into sentences
                sentences = self._split_into_sentences(
                    chunk_text.strip(),
                    paragraph_id,
                    section.section_id,
                    document.document_id,
                    sentence_id,
                )
                paragraph.sentences = sentences
                sentence_id += len(sentences)

                section.paragraphs.append(paragraph)
                paragraph_id += 1

            # Only add section if it has content
            if section.paragraphs:
                # Set section content_text from paragraphs
                section.content_text = "\n\n".join(p.text for p in section.paragraphs)
                document.sections.append(section)
            section_id += 1

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Intelligent paragraph splitting that handles various text formats."""
        # First try double newline splitting
        chunks = text.split("\n\n")

        # If that results in very few chunks, try other strategies
        if len(chunks) <= 2 and len(text) > 1000:
            # Try splitting on single newlines followed by capital letters
            import re

            alt_chunks = re.split(r"\n(?=[A-Z])", text)
            if len(alt_chunks) > len(chunks) * 2:
                chunks = alt_chunks

        # Clean and filter chunks
        paragraphs = []
        for chunk in chunks:
            cleaned = chunk.strip()
            if cleaned and len(cleaned) > 20:  # Minimum paragraph length
                paragraphs.append(cleaned)

        return paragraphs

    def _split_by_headers(self, markdown_text: str) -> List[tuple]:
        """Split markdown text by headers."""
        lines = markdown_text.split("\n")
        sections = []
        current_title = None
        current_content = []

        for line in lines:
            # Check if line is a header
            if line.strip().startswith("#"):
                # Save previous section
                if current_title is not None or current_content:
                    sections.append((current_title, "\n".join(current_content)))

                # Start new section
                current_title = line.strip().lstrip("#").strip()
                current_content = []
            else:
                current_content.append(line)

        # Add final section
        if current_title is not None or current_content:
            sections.append((current_title, "\n".join(current_content)))

        return sections

    def _split_into_sentences(
        self,
        text: str,
        paragraph_id: Union[int, str],
        section_id: Union[int, str],
        document_id: Union[int, str],
        start_sentence_id: int,
    ) -> List[DocumentSentence]:
        """Split text into sentences."""
        if self.sentence_splitter:
            sentence_chunks = self.sentence_splitter.split_text(text)
        else:
            # Simple sentence splitting
            import re

            sentence_chunks = re.split(r"(?<=[.!?])\s+", text)

        sentences = []
        for i, sentence_text in enumerate(sentence_chunks):
            if not sentence_text.strip():
                continue

            sentence = DocumentSentence(
                text=sentence_text.strip(),
                sentence_id=str(start_sentence_id + i),
                paragraph_id=str(paragraph_id),
                section_id=str(section_id),
                document_id=str(document_id),
                sentence_index=i,
            )
            sentences.append(sentence)

        return sentences

    def _extract_images(self, docling_doc, document: Document):
        """Extract images from the document."""
        try:
            images = []

            # Look for pictures in page elements
            if hasattr(docling_doc, "pictures"):
                for picture in docling_doc.pictures:
                    if (
                        hasattr(picture, "label")
                        and "picture" in str(picture.label).lower()
                    ):
                        try:
                            # Get the provenance items from the picture
                            page_num = None
                            bbox = None
                            provenance_items = picture.prov
                            for provenance_item in provenance_items:
                                page_num = provenance_item.page_no
                                bbox = provenance_item.bbox
                                break

                            image_data = {
                                "type": "image",
                                "page": page_num,
                                "bbox": bbox.model_dump() if bbox else {},
                                "element_type": str(picture.label),
                            }
                            images.append(image_data)
                        except Exception:
                            pass

            document.metadata.images = images

        except Exception:
            pass

    def _extract_tables(self, docling_doc, document: Document):
        """Extract table data from the document."""
        try:
            tables = []

            if hasattr(docling_doc, "tables"):
                for table in docling_doc.tables:
                    if hasattr(table, "label") and "table" in str(table.label).lower():
                        try:
                            # Get the provenance items from the table
                            provenance_items = table.prov
                            page_num = None
                            bbox = None
                            for provenance_item in provenance_items:
                                page_num = provenance_item.page_no
                                bbox = provenance_item.bbox
                                break

                            table_data = {
                                "type": "table",
                                "page": page_num,
                                "element_type": str(table.label),
                                "text": table.export_to_markdown(docling_doc),
                                "bbox": bbox.model_dump() if bbox else {},
                            }
                            tables.append(table_data)
                        except Exception:
                            pass

            document.metadata.tables = tables

        except Exception:
            pass

    def _extract_enhanced_metadata(self, document: Document):
        """Extract enhanced metadata using LLM with intelligent text selection and retry logic."""
        try:
            import unify

            client = unify.Unify(
                "o4-mini@openai",
                cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            )

            # Build extraction prompt with schema
            from unity.file_manager.parser.prompt_builders import (
                build_metadata_extraction_prompt,
            )

            prompt = build_metadata_extraction_prompt()

            # Define text sources in priority order
            text_sources = []

            # Priority 1: Full document text (if available and not too large)
            if hasattr(document, "full_text") and document.full_text:
                # Use full text if it's reasonable size (up to ~16k chars for good results)
                if len(document.full_text) <= 16000:
                    text_sources.append(("full_text", document.full_text))
                else:
                    # Take beginning, middle, and end portions for comprehensive coverage
                    text_length = len(document.full_text)
                    chunk_size = 5000
                    beginning = document.full_text[:chunk_size]
                    middle_start = (text_length // 2) - (chunk_size // 2)
                    middle = document.full_text[
                        middle_start : middle_start + chunk_size
                    ]
                    end = document.full_text[-chunk_size:]
                    combined = f"{beginning}\n\n[...Document middle section...]\n\n{middle}\n\n[...Document end section...]\n\n{end}"
                    text_sources.append(("full_text_sampled", combined))

            # Priority 2: Document summary (if available)
            if document.summary and len(document.summary) > 100:
                summary_text = document.summary
                # Add some section summaries for context
                if document.sections:
                    for section in document.sections[:2]:
                        if section.summary:
                            summary_text += f"\n\nSection: {section.title or 'Content'}\n{section.summary}"
                text_sources.append(("document_summary", summary_text))

            # Priority 3: All section summaries
            if document.sections:
                section_summaries = []
                for i, section in enumerate(document.sections):
                    if section.summary:
                        section_summaries.append(
                            f"Section {i+1}: {section.title or 'Content'}\n{section.summary}",
                        )
                if section_summaries:
                    text_sources.append(
                        ("section_summaries", "\n\n".join(section_summaries[:5])),
                    )

            # Priority 4: First few paragraphs from each section
            if document.sections:
                paragraph_samples = []
                for i, section in enumerate(document.sections[:3]):
                    if section.paragraphs:
                        section_text = f"Section {i+1}: {section.title or 'Content'}\n"
                        # Get first 2 paragraphs from each section
                        for j, para in enumerate(section.paragraphs[:2]):
                            para_text = para.summary or para.text
                            section_text += f"{para_text[:500]}\n"
                        paragraph_samples.append(section_text)
                if paragraph_samples:
                    text_sources.append(
                        ("paragraph_samples", "\n\n".join(paragraph_samples)),
                    )

            # Priority 5: Raw text fallback
            if hasattr(document, "content") and document.content:
                text_sources.append(("raw_content", document.content[:4000]))

            # Try each text source in priority order
            last_error = None
            for source_name, metadata_text in text_sources:
                if not metadata_text or len(metadata_text.strip()) < 50:
                    continue

                try:
                    # Ensure reasonable size
                    if len(metadata_text) > 16000:
                        metadata_text = (
                            metadata_text[:16000]
                            + "\n[Text truncated for processing...]"
                        )

                    # Make the API call with retries for transient errors
                    response = None
                    for attempt in range(3):
                        try:
                            response = client.copy().generate(prompt + metadata_text)
                            break
                        except Exception as api_error:
                            if attempt < 2 and "rate" in str(api_error).lower():
                                import time

                                time.sleep(2**attempt)  # Exponential backoff
                                continue
                            raise

                    if not response:
                        continue

                    # Parse and validate response
                    try:
                        validated_metadata = (
                            DocumentMetadataExtraction.model_validate_json(response)
                        )

                        # Successfully parsed - update document metadata
                        document.metadata.document_type = (
                            validated_metadata.document_type
                        )
                        document.metadata.category = validated_metadata.category
                        document.metadata.key_topics = validated_metadata.key_topics
                        document.metadata.named_entities = (
                            validated_metadata.named_entities
                        )
                        document.metadata.content_tags = validated_metadata.content_tags
                        document.metadata.confidence_score = (
                            validated_metadata.confidence_score
                        )

                        return  # Success - exit the method

                    except Exception as parse_error:

                        last_error = parse_error
                        continue

                except Exception as extraction_error:

                    last_error = extraction_error
                    continue

            # If we get here, all sources failed

        except Exception as e:
            pass

    def _generate_summaries(self, document: Document):
        """Generate hierarchical summaries using parallel map-reduce approach."""

        try:
            import unify
            from unity.file_manager.parser.prompt_builders import (
                build_paragraph_summary_prompt,
                build_section_summary_prompt,
                build_document_summary_prompt,
                build_chunked_text_summary_prompt,
            )

            # Create a single client for the entire summarization process
            client = unify.Unify(
                "o4-mini@openai",
                cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            )

            # Industry-standard token limits based on model capabilities
            # Using tiktoken's cl100k_base tokenizer estimates (GPT-3.5/4)
            # Average ratio: ~1 token per 4 characters for English text
            # Model context windows: GPT-3.5 (16k), GPT-4 (8k-128k)
            # Reserve ~2k tokens for prompt and response formatting

            # Start with generous limit, progressively reduce if needed
            INITIAL_MAX_TOKENS = 6000  # ~24k chars
            TOKEN_REDUCTION_STEP = 1500  # Reduce by ~6k chars each iteration
            MIN_TOKEN_LIMIT = 1500  # ~6k chars minimum
            MAX_RETRY_ATTEMPTS = 3

            # Chunking overlap for context preservation (20% overlap is standard)
            CHUNK_OVERLAP_RATIO = 0.2

            def _estimate_tokens(text: str) -> int:
                """Estimate token count using cl100k_base approximation."""
                # Industry standard: ~4 characters per token for English
                # Add 10% buffer for safety
                return int(len(text) / 4 * 1.1)

            def _semantic_chunk_text(text: str, max_chars: int) -> List[str]:
                """Split text into semantic chunks with overlap, respecting sentence boundaries."""
                if len(text) <= max_chars:
                    return [text]

                # Split by sentences first (preserve semantic units)
                import re

                sentences = re.split(r"(?<=[.!?])\s+", text)

                chunks = []
                current_chunk = []
                current_length = 0
                overlap_buffer = []
                overlap_size = int(max_chars * CHUNK_OVERLAP_RATIO)

                for sentence in sentences:
                    sentence_length = len(sentence) + 1  # +1 for space

                    # If single sentence exceeds limit, split by clauses
                    if sentence_length > max_chars:
                        # Split by common clause markers
                        clauses = re.split(r"(?<=[,;:])\s+", sentence)
                        for clause in clauses:
                            if current_length + len(clause) > max_chars:
                                if current_chunk:
                                    chunks.append(" ".join(current_chunk))
                                    # Keep last portion for overlap
                                    overlap_buffer = (
                                        current_chunk[-3:]
                                        if len(current_chunk) > 3
                                        else current_chunk
                                    )
                                    current_chunk = overlap_buffer + [clause]
                                    current_length = sum(
                                        len(c) + 1 for c in current_chunk
                                    )
                                else:
                                    # Force add even if too long
                                    chunks.append(clause)
                            else:
                                current_chunk.append(clause)
                                current_length += len(clause) + 1
                    elif current_length + sentence_length > max_chars:
                        # Complete current chunk
                        if current_chunk:
                            chunks.append(" ".join(current_chunk))
                            # Create overlap from last sentences
                            overlap_text = " ".join(current_chunk)
                            if len(overlap_text) > overlap_size:
                                # Take last portion of text for overlap
                                overlap_start = len(overlap_text) - overlap_size
                                # Find sentence boundary in overlap region
                                overlap_match = re.search(
                                    r"[.!?]\s+",
                                    overlap_text[overlap_start:],
                                )
                                if overlap_match:
                                    overlap_start += overlap_match.end()
                                overlap_buffer = overlap_text[overlap_start:].split()
                            else:
                                overlap_buffer = (
                                    current_chunk[-2:]
                                    if len(current_chunk) > 2
                                    else current_chunk
                                )

                        current_chunk = overlap_buffer + [sentence]
                        current_length = sum(len(s) + 1 for s in current_chunk)
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length

                # Add final chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))

                return chunks

            def _chunk_text_if_needed(
                text: str,
                prompt_builder,
                context_info: dict = None,
                max_tokens: int = INITIAL_MAX_TOKENS,
                attempt: int = 0,
            ):
                """Split text into chunks if needed, with progressive token limit reduction."""

                max_chars = max_tokens * 4  # Convert tokens to approximate characters

                # Build the prompt first to account for its tokens
                if context_info and "chunk_number" in context_info:
                    prompt = prompt_builder(
                        context_info["chunk_number"],
                        context_info["total_chunks"],
                    )
                else:
                    prompt = prompt_builder()

                prompt_tokens = _estimate_tokens(prompt)
                available_chars = max(
                    (max_tokens - prompt_tokens) * 4,
                    1000,
                )  # Minimum 1000 chars

                if len(text) <= available_chars:
                    # Text fits in single call
                    try:
                        return client.copy().generate(prompt + text).strip()
                    except Exception as e:
                        # If we hit token limit, retry with reduced limit
                        if attempt < MAX_RETRY_ATTEMPTS and "token" in str(e).lower():
                            new_max_tokens = max(
                                max_tokens - TOKEN_REDUCTION_STEP,
                                MIN_TOKEN_LIMIT,
                            )
                            return _chunk_text_if_needed(
                                text,
                                prompt_builder,
                                context_info,
                                new_max_tokens,
                                attempt + 1,
                            )
                        elif attempt >= MAX_RETRY_ATTEMPTS:
                            # Final fallback: return truncated text
                            return (
                                text[:available_chars]
                                + "... [Text truncated due to length]"
                            )
                        else:
                            raise

                # Text needs chunking
                chunks = _semantic_chunk_text(text, available_chars)

                if len(chunks) == 1:
                    # Single chunk after semantic splitting
                    try:
                        return client.copy().generate(prompt + chunks[0]).strip()
                    except Exception as e:
                        if attempt < MAX_RETRY_ATTEMPTS and "token" in str(e).lower():
                            new_max_tokens = max(
                                max_tokens - TOKEN_REDUCTION_STEP,
                                MIN_TOKEN_LIMIT,
                            )
                            return _chunk_text_if_needed(
                                text,
                                prompt_builder,
                                context_info,
                                new_max_tokens,
                                attempt + 1,
                            )
                        else:
                            return (
                                chunks[0][:available_chars]
                                + "... [Text truncated due to length]"
                            )

                # Summarize chunks in parallel

                # Prepare chunk data for parallel processing
                chunk_data = []
                for i, chunk in enumerate(chunks):
                    chunk_data.append(
                        {
                            "chunk": chunk,
                            "chunk_num": i + 1,
                            "total_chunks": len(chunks),
                        },
                    )

                def _summarize_chunk(**data):
                    """Summarize a single chunk."""
                    chunk_prompt = build_chunked_text_summary_prompt(
                        data["chunk_num"],
                        data["total_chunks"],
                    )
                    try:
                        return (
                            client.copy().generate(chunk_prompt + data["chunk"]).strip()
                        )
                    except Exception as e:
                        if "token" in str(e).lower() and attempt < MAX_RETRY_ATTEMPTS:
                            # Retry with reduced tokens
                            new_max_tokens = max(
                                max_tokens - TOKEN_REDUCTION_STEP,
                                MIN_TOKEN_LIMIT,
                            )
                            return _chunk_text_if_needed(
                                data["chunk"],
                                lambda: build_chunked_text_summary_prompt(
                                    data["chunk_num"],
                                    data["total_chunks"],
                                ),
                                None,
                                new_max_tokens,
                                attempt + 1,
                            )
                        else:
                            # Fallback
                            return data["chunk"][:1000] + "... [Chunk truncated]"

                # Run chunk summaries in parallel
                chunk_summaries = unify.map(
                    _summarize_chunk,
                    chunk_data,
                    name=f"Chunk Summaries ({len(chunks)} chunks)",
                )

                # Combine chunk summaries
                if len(chunk_summaries) > 1:
                    combined = "\n\n".join(chunk_summaries)
                    # Check if combined summaries need further summarization
                    if _estimate_tokens(combined) > max_tokens:
                        # Recursive summarization
                        return _chunk_text_if_needed(
                            combined,
                            prompt_builder,
                            context_info,
                            max_tokens,
                            attempt,
                        )
                    else:
                        try:
                            return (
                                client.copy()
                                .generate(prompt_builder() + combined)
                                .strip()
                            )
                        except Exception:
                            return combined
                else:
                    return chunk_summaries[0]

            # LEVEL 1: Parallel Paragraph summaries

            # Collect all paragraphs that need summarization
            paragraphs_to_process = []
            paragraph_refs = []  # Keep track of (section_idx, para_idx) for updating

            for section_idx, section in enumerate(document.sections):
                for para_idx, para in enumerate(section.paragraphs):
                    if para.summary is None:
                        # Always generate summaries (no short text bypass)
                        # This ensures we get topics, entities, etc. even for short text
                        paragraphs_to_process.append(
                            {
                                "text": para.text,
                                "section_idx": section_idx,
                                "para_idx": para_idx,
                                "section_title": section.title,
                            },
                        )
                        paragraph_refs.append((section_idx, para_idx))

            total_paragraphs = len(paragraphs_to_process)

            # Define the paragraph summary runner
            def _summarize_paragraph(**para_data):
                """Summarize a single paragraph."""
                return _chunk_text_if_needed(
                    para_data["text"],
                    build_paragraph_summary_prompt,
                )

            # Run all paragraph summaries in parallel
            if paragraphs_to_process:
                para_summaries = unify.map(
                    _summarize_paragraph,
                    paragraphs_to_process,
                    name="Paragraph Summaries",
                )

                # Update the document with the summaries
                for (section_idx, para_idx), summary in zip(
                    paragraph_refs,
                    para_summaries,
                ):
                    document.sections[section_idx].paragraphs[
                        para_idx
                    ].summary = summary

            # LEVEL 2: Parallel Section summaries

            # Collect all sections that need summarization
            sections_to_process = []
            section_refs = []

            for section_idx, section in enumerate(document.sections):
                if section.summary is None and section.paragraphs:
                    # Combine paragraph summaries
                    para_summaries = []
                    for para in section.paragraphs:
                        if para.summary:
                            para_summaries.append(
                                f"Paragraph {len(para_summaries) + 1}:\n{para.summary}",
                            )

                    if para_summaries:
                        combined_summaries = "\n\n".join(para_summaries)

                        # Add section title for context
                        if section.title:
                            combined_summaries = f"Section Title: {section.title}\n\n{combined_summaries}"

                        # Always generate section summaries (no short text bypass)
                        sections_to_process.append(
                            {
                                "combined_summaries": combined_summaries,
                                "section_idx": section_idx,
                                "section_title": section.title,
                                "num_paragraphs": len(para_summaries),
                            },
                        )
                        section_refs.append(section_idx)

                    # Store combined full text for reference
                    section.content_text = "\n\n".join(
                        p.text for p in section.paragraphs
                    )

            total_sections = len(sections_to_process)

            # Define the section summary runner
            def _summarize_section(**section_data):
                """Summarize a single section."""
                return _chunk_text_if_needed(
                    section_data["combined_summaries"],
                    build_section_summary_prompt,
                )

            # Run all section summaries in parallel
            if sections_to_process:
                section_summaries = unify.map(
                    _summarize_section,
                    sections_to_process,
                    name="Section Summaries",
                )

                # Update the document with the summaries
                for section_idx, summary in zip(section_refs, section_summaries):
                    document.sections[section_idx].summary = summary

            # LEVEL 3: Document summary (from section summaries)
            if document.summary is None and document.sections:
                section_summaries = []
                for idx, section in enumerate(document.sections):
                    if section.summary:
                        section_info = f"Section {idx + 1}: {section.title or 'Content'}\n{section.summary}"
                        section_summaries.append(section_info)
                    elif section.title:
                        # Use title if no summary available
                        section_summaries.append(f"Section {idx + 1}: {section.title}")

                if section_summaries:
                    combined_sections = "\n\n".join(section_summaries)

                    # Add document metadata for context
                    doc_context = f"Document: {document.metadata.title or 'Untitled'}\n"
                    doc_context += f"Type: {document.metadata.file_type}\n"
                    doc_context += f"Total Sections: {len(document.sections)}\n\n"

                    # Generate document summary from section summaries
                    document.summary = _chunk_text_if_needed(
                        doc_context + combined_sections,
                        build_document_summary_prompt,
                    )

            print("Summary generation completed")

        except Exception as e:
            # Fallback to basic summaries
            self._generate_basic_summaries(document)

    def _generate_basic_summaries(self, document: Document):
        """Generate basic summaries as fallback when LLM summarization fails."""
        try:
            # Basic paragraph summaries - first 200 chars
            for section in document.sections:
                for para in section.paragraphs:
                    if para.summary is None and para.text:
                        para.summary = (
                            para.text[:200] + "..."
                            if len(para.text) > 200
                            else para.text
                        )

            # Basic section summaries - combine paragraph starts
            for section in document.sections:
                if section.summary is None and section.paragraphs:
                    summary_parts = []
                    for i, para in enumerate(
                        section.paragraphs[:3],
                    ):  # First 3 paragraphs
                        text = para.summary or para.text[:100]
                        summary_parts.append(text)
                    section.summary = " ".join(summary_parts)

            # Basic document summary - combine section titles and starts
            if document.summary is None and document.sections:
                summary_parts = []
                for section in document.sections[:5]:  # First 5 sections
                    if section.title:
                        summary_parts.append(
                            f"{section.title}: {(section.summary or '')[:100]}",
                        )
                    elif section.summary:
                        summary_parts.append(section.summary[:100])
                document.summary = " ".join(summary_parts)

        except Exception:
            # Ultimate fallback
            document.summary = (
                document.full_text[:500]
                if document.full_text
                else "Document parsing completed."
            )

    def _update_statistics(self, document: Document):
        """Update document statistics."""
        document.metadata.total_sections = len(document.sections)
        document.metadata.total_paragraphs = document.get_total_paragraphs()
        document.metadata.total_sentences = document.get_total_sentences()

        # Update metadata for sections
        for section in document.sections:
            section.metadata["paragraph_count"] = len(section.paragraphs)
            section.metadata["sentence_count"] = sum(
                len(p.sentences) for p in section.paragraphs
            )
