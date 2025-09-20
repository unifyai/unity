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
import logging
import hashlib
from typing import Any, Dict, List, Union

from .base import GenericParser
from .converter import DocumentConversionManager, DocxToPdfConverter
from .types.document import (
    Document,
    DocumentMetadata,
    DocumentMetadataExtraction,
    DocumentParagraph,
    DocumentSection,
    DocumentSentence,
    DocumentImage,
    DocumentTable,
)
from .token_utils import (
    count_tokens_per_utf_byte,
    has_meaningful_text,
)
from .summary_utils import generate_summary_with_compression
from .token_utils import (
    is_within_token_limit_bytes,
    clip_text_to_token_limit_bytes,
    first_tokens_per_utf_byte,
    middle_tokens_per_utf_byte,
    last_tokens_per_utf_byte,
    conservative_token_estimate,
    is_within_token_limit_conservative,
    clip_text_to_token_limit_conservative,
)
from .prompt_builders import build_picture_description_prompt
from ...common.llm_helpers import short_id

# Check for optional dependencies
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        PictureDescriptionVlmOptions,
    )
    from docling_core.types.doc.document import PictureDescriptionData
    from docling.chunking import HybridChunker
    from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
    import tiktoken

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

    # Placeholder classes for when Docling is not available
    class HybridChunker:
        pass

    class OpenAITokenizer:
        pass


try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    RecursiveCharacterTextSplitter = None

# Optional spaCy (lightweight) for robust sentence splitting
try:
    import spacy  # type: ignore

    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False
    spacy = None  # type: ignore


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
        conversion_parallel: bool = False,
        cleanup_converted_files: bool = True,
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
        self.conversion_parallel = conversion_parallel
        self.cleanup_converted_files = cleanup_converted_files

        # File format conversion manager: register available converters (extensible)
        self._conversion_manager = DocumentConversionManager(
            converters=[DocxToPdfConverter()],
        )

        # Initialize Docling converter (PDF pipeline) if available
        self.converter = None
        try:
            PICTURE_DESCRIPTION_MODEL_REPO = os.environ.get(
                "PICTURE_DESCRIPTION_MODEL_REPO",
                "HuggingFaceTB/SmolVLM-500M-Instruct",
            )
        except Exception:
            PICTURE_DESCRIPTION_MODEL_REPO = "HuggingFaceTB/SmolVLM-500M-Instruct"

        if DOCLING_AVAILABLE:
            pipeline_options = PdfPipelineOptions()
            picture_description_options = PictureDescriptionVlmOptions(
                repo_id=PICTURE_DESCRIPTION_MODEL_REPO,
                prompt=build_picture_description_prompt(),
            )
            pipeline_options.do_picture_description = True
            pipeline_options.picture_description_options = picture_description_options
            pipeline_options.images_scale = 2.0
            pipeline_options.generate_picture_images = True

            converter = DocumentConverter(
                allowed_formats=list(InputFormat),
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                },
            )
            self.converter = converter

        else:
            logging.warning(
                "Docling modules not available; hybrid parsing and rich extraction are disabled. Falling back to basic parsing.",
            )

        # Initialize spaCy (optional) and text splitters
        self._spacy_nlp: Language | None = None
        if SPACY_AVAILABLE:
            self._init_spacy_pipeline()
        else:
            logging.warning(
                "spaCy not available; sentence splitting will use LangChain/regex.",
            )

        # Initialize text splitters (overridden below once chunker is ready)
        self.paragraph_splitter = None
        self.sentence_splitter = None
        if LANGCHAIN_AVAILABLE:
            # Baseline initializers; will be replaced if hybrid chunker is enabled
            # Sensible defaults from common guidance: ~10% overlap with caps
            para_overlap_chars = min(200, max(0, int(0.1 * max_chunk_size)))
            sent_overlap_chars = min(16, max(0, int(0.1 * sentence_chunk_size)))

            self.paragraph_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=para_overlap_chars,
                length_function=len,
                separators=["\n\n", "\n", " ", ""],
                keep_separator=True,
            )
            self.sentence_splitter = RecursiveCharacterTextSplitter(
                chunk_size=sentence_chunk_size,
                chunk_overlap=sent_overlap_chars,
                length_function=len,
                # Prefer sentence boundaries; keep punctuation with the preceding chunk
                separators=[". ", "! ", "? ", "\n\n", "\n"],
                keep_separator="end",
            )
        else:
            logging.warning(
                "LangChain text splitters not available; falling back to regex-based splitting.",
            )

        # Initialize hybrid chunker if requested
        self.hybrid_chunker = None
        if self.use_hybrid_chunking:
            self._init_hybrid_chunking()

        # If hybrid chunker and LangChain are available, align splitters with tokenizer limits
        if LANGCHAIN_AVAILABLE and self.hybrid_chunker is not None:
            try:
                import tiktoken

                try:
                    EMBEDDING_MODEL = os.environ.get(
                        "EMBEDDING_MODEL",
                        "text-embedding-3-small",
                    )
                except Exception:
                    EMBEDDING_MODEL = "text-embedding-3-small"

                try:
                    EMBEDDING_ENCODING = os.environ.get(
                        "EMBEDDING_ENCODING",
                        "cl100k_base",
                    )
                except Exception:
                    EMBEDDING_ENCODING = "cl100k_base"

                encoding_name = tiktoken.encoding_for_model(EMBEDDING_MODEL).name
            except Exception:
                encoding_name = "cl100k_base"

            # Paragraph splitter: token-aware, aligned to chunker's token budget
            try:
                para_overlap_tokens = min(
                    256,
                    max(0, int(0.1 * self.hybrid_chunker.max_tokens)),
                )
                self.paragraph_splitter = (
                    RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                        encoding_name=encoding_name,
                        chunk_size=self.hybrid_chunker.max_tokens,
                        chunk_overlap=para_overlap_tokens,
                        separators=["\n\n", "\n", " "],
                        keep_separator=True,
                    )
                )
            except Exception:
                logging.warning(
                    "Failed to initialize token-aware paragraph splitter; using baseline splitter.",
                )

            # Sentence splitter: token-aware, sentence boundaries, keep punctuation with previous
            try:
                # Keep sentence size modest to encourage one-sentence-per-chunk behaviour
                target_sentence_tokens = max(
                    64,
                    min(self.sentence_chunk_size, self.hybrid_chunker.max_tokens // 4),
                )
                sent_overlap_tokens = min(16, max(0, int(0.1 * target_sentence_tokens)))
                self.sentence_splitter = (
                    RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                        encoding_name=encoding_name,
                        chunk_size=target_sentence_tokens,
                        chunk_overlap=sent_overlap_tokens,
                        separators=[". ", "! ", "? ", "\n\n", "\n"],
                        keep_separator="end",
                    )
                )
            except Exception:
                logging.warning(
                    "Failed to initialize token-aware sentence splitter; using baseline splitter.",
                )

        # Supported formats
        self.supported_formats = self._get_supported_formats()

    def _init_spacy_pipeline(self) -> None:
        """Initialise a minimal spaCy pipeline for sentence segmentation.

        Tries to load en_core_web_sm if present; otherwise falls back to blank('en')
        with rule-based sentencizer. Attempts auto-download if allowed.
        """
        if not SPACY_AVAILABLE:
            return
        model = os.environ.get("SPACY_MODEL", "en_core_web_sm")
        try:
            # Load small English model with minimal components
            self._spacy_nlp = spacy.load(model)
            # Ensure we have sentence boundaries
            if (
                "senter" not in self._spacy_nlp.pipe_names
                and "sentencizer" not in self._spacy_nlp.pipe_names
            ):
                self._spacy_nlp.add_pipe("sentencizer")
            # Disable heavy components not needed for sentence splitting
            for comp in ("parser", "attribute_ruler", "tagger", "lemmatizer", "ner"):
                try:
                    if comp in self._spacy_nlp.pipe_names:
                        self._spacy_nlp.remove_pipe(comp)
                except Exception:
                    pass
        except Exception as e:
            # Do not attempt programmatic downloads or fallbacks; re-raise exact exception
            raise e

        # Add a lightweight post-processor to prevent enum prefixes becoming their own sentences
        try:

            if self._spacy_nlp is not None:
                after_component = (
                    "sentencizer"
                    if "sentencizer" in self._spacy_nlp.pipe_names
                    else ("senter" if "senter" in self._spacy_nlp.pipe_names else None)
                )
                if after_component:
                    self._spacy_nlp.add_pipe("sent_fix_enums", after=after_component)
                else:
                    # If no sentence component present, add fix last (no-op until boundaries exist)
                    self._spacy_nlp.add_pipe("sent_fix_enums", last=True)
        except Exception:
            # Non-fatal: fallback behaviour remains
            pass

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
            logging.warning(
                "Docling backend unavailable; only basic text formats are supported.",
            )
            return [".txt"]

    def _compute_document_id(
        self,
        path: Path | None,
        *,
        full_text: str | None = None,
    ) -> str:
        """Compute a stable, content-based document ID.

        Preference order:
        1) File bytes hash (sha256) if path is available and readable
        2) Full text hash (sha256) if provided
        3) Fallback to a deterministic hash of the stringified path
        """
        try:
            if (
                path is not None
                and isinstance(path, Path)
                and path.exists()
                and path.is_file()
            ):
                with path.open("rb") as f:
                    data = f.read()
                return hashlib.sha256(data).hexdigest()
        except Exception:
            pass

        try:
            if isinstance(full_text, str) and full_text:
                return hashlib.sha256(
                    full_text.encode("utf-8", errors="ignore"),
                ).hexdigest()
        except Exception:
            pass

        # Last resort (path string, may vary across machines but deterministic per input)
        key = str(path) if path is not None else "unknown"
        return hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()

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

    def _index_docling_structure(self, docling_doc) -> dict:
        """Build an index of headings and item → section-path mapping from Docling's tree.

        Returns a dict with keys:
        - ref_to_path: map of NodeItem.self_ref -> tuple[str, ...] headings path
        - heading_refs: set of self_refs that are SectionHeaderItems (or Title)
        - heading_ref_to_level: map of heading self_ref -> explicit Docling level (int)
        - heading_order: list of {text, level, self_ref, path} in document order
        - title: optional top-level title text if present
        """
        index: dict = {
            "ref_to_path": {},
            "heading_refs": set(),
            "heading_ref_to_level": {},
            "heading_order": [],
            "title": None,
        }

        # Maintain a stack of (text, level, self_ref) for section headers
        heading_stack: list[tuple[str, int, str]] = []

        try:
            iterator = docling_doc.iterate_items(
                with_groups=True,
                traverse_pictures=False,
            )
        except Exception:
            # Fallback: no index available
            return index

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
                    header_level = (
                        int(raw_level)
                        if isinstance(raw_level, int)
                        else (1 if is_title else 1)
                    )

                    # Maintain a proper stack (levels are 1-based)
                    while heading_stack and heading_stack[-1][1] >= header_level:
                        heading_stack.pop()
                    if self_ref is None:
                        # If no self_ref, fabricate a stable key from text (rare)
                        self_ref = f"#/_heading/{len(index['heading_order'])}"

                    heading_stack.append(
                        (text_val or "Untitled", header_level, self_ref),
                    )

                    # Current path
                    path = tuple(h[0] for h in heading_stack)

                    index["heading_refs"].add(self_ref)
                    index["heading_ref_to_level"][self_ref] = header_level
                    index["ref_to_path"][self_ref] = path
                    index["heading_order"].append(
                        {
                            "text": text_val,
                            "level": header_level,
                            "self_ref": self_ref,
                            "path": list(path),
                        },
                    )

                    if is_title and text_val and not index.get("title"):
                        index["title"] = text_val

                # Map every item's self_ref to the current path (if any)
                if self_ref is not None:
                    if heading_stack:
                        index["ref_to_path"][self_ref] = tuple(
                            h[0] for h in heading_stack
                        )
            except Exception:
                continue

        return index

    def _init_hybrid_chunking(self) -> bool:
        """Initialize the hybrid chunker if available."""
        if not DOCLING_AVAILABLE or not LANGCHAIN_AVAILABLE:
            if not DOCLING_AVAILABLE:
                logging.warning(
                    "Hybrid chunking requires Docling; skipping hybrid initialization.",
                )
            if not LANGCHAIN_AVAILABLE:
                logging.warning(
                    "Hybrid chunking benefits from LangChain; token-aware splitters will be unavailable.",
                )
            return False

        try:
            try:
                EMBEDDING_MODEL = os.environ.get(
                    "EMBEDDING_MODEL",
                    "text-embedding-3-small",
                )
            except Exception:
                EMBEDDING_MODEL = "text-embedding-3-small"

            try:
                EMBEDDING_MAX_INPUT_TOKENS = int(
                    os.environ.get("EMBEDDING_MAX_INPUT_TOKENS", "8000"),
                )
            except Exception:
                EMBEDDING_MAX_INPUT_TOKENS = 8000

            # Use a default embedding model tokenizer for chunking
            # Following Docling's best practices from documentation
            tokenizer = OpenAITokenizer(
                tokenizer=tiktoken.encoding_for_model(EMBEDDING_MODEL),
                max_tokens=EMBEDDING_MAX_INPUT_TOKENS,
            )

            # Initialize hybrid chunker with best practice settings
            self.hybrid_chunker = HybridChunker(
                tokenizer=tokenizer,
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

        # Pre-conversion: convert inputs for supported formats (e.g., .doc/.docx -> .pdf)
        converted_paths, to_cleanup = self._maybe_convert_inputs([file_path])
        file_path = converted_paths[0]

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

        # Save parsed result if enabled
        self._save_parsed_result_if_enabled(file_path, document)

        # Cleanup converted artifacts if enabled
        if self.cleanup_converted_files and to_cleanup:
            self._cleanup_files(to_cleanup)

        return document

    def parse_batch(
        self,
        file_paths: List[Union[str, Path]],
        /,
        *,
        raises_on_error: bool = False,
        parallelize_post: bool = True,
        **options: Any,
    ) -> List[Document]:
        """
        Batch-parse multiple documents.

        Behaviour:
        - Uses Docling's convert_all when available to batch convert inputs efficiently.
        - Falls back to per-file parse for unsupported formats or when Docling is unavailable.
        - Post-processing (structure extraction, summaries, metadata) is parallelized via unify.map
          when available, with a sequential fallback.
        """
        normalized: List[Path] = [Path(p).expanduser().resolve() for p in file_paths]

        # Pre-conversion pass: convert supported formats (preserve order), allow safe parallelism
        normalized, to_cleanup = self._maybe_convert_inputs(
            normalized,
            parallel=self.conversion_parallel,
        )

        # Continue with Docling's convert_all / per-file parse
        documents: List[Document] = []

        # Helper to run a function with optional parallelism
        def _safe_map(name: str, items: list[dict], fn):
            try:
                import unify

                return unify.map(fn, items, name=name) if items else []
            except Exception:
                return [fn(**it) for it in items]

        # Fast path: Use convert_all only for formats Docling handles and when converter is present
        can_batch = (
            DOCLING_AVAILABLE
            and self.converter is not None
            and all(
                p.suffix.lower() in self.supported_formats and p.is_file()
                for p in normalized
            )
        )

        if not can_batch:
            # Fallback to individual parse preserving existing logic
            for p in normalized:
                try:
                    documents.append(self.parse(str(p), **options))
                except Exception as e:
                    if raises_on_error:
                        raise
                    # Create minimal failed document with metadata
                    try:
                        doc_id = short_id(4)
                    except Exception:
                        doc_id = self._compute_document_id(p)[:8]
                    meta = self._create_base_metadata(p)
                    documents.append(
                        Document(
                            document_id=doc_id,
                            metadata=meta,
                            full_text="",
                            processing_status="failed",
                        ),
                    )
            return documents

        # Split supported/unsupported for batch conversion while preserving order mapping
        supported_indices: list[int] = []
        unsupported_indices: list[int] = []
        supported_paths: list[Path] = []
        for idx, p in enumerate(normalized):
            if p.suffix.lower() in self.supported_formats and p.is_file():
                supported_indices.append(idx)
                supported_paths.append(p)
            else:
                unsupported_indices.append(idx)

        start_time = time.time()
        conv_results = self.converter.convert_all(
            [str(p) for p in supported_paths],
            raises_on_error=raises_on_error,
        )

        # Prepare post-processing tasks for supported inputs
        tasks: list[dict] = []
        for conv_res, src in zip(conv_results, supported_paths):
            tasks.append(
                {
                    "conv_res": conv_res,
                    "src_path": src,
                    "options": options,
                    "start_time": start_time,
                },
            )

        def _post_process(**data) -> Document:
            conv_res = data["conv_res"]
            src_path = data.get("src_path")
            opts = data.get("options", {})

            if conv_res.status != ConversionStatus.SUCCESS:
                # Fallback: basic parse to keep behaviour consistent
                try:
                    if src_path and src_path.suffix.lower() in [
                        ".txt",
                        ".log",
                        ".json",
                    ]:
                        return self._parse_as_text_document(src_path, **opts)
                except Exception:
                    pass
                # Minimal document for failure
                doc_id = (
                    self._compute_document_id(src_path)
                    if src_path
                    else str(time.time())
                )
                meta = (
                    self._create_base_metadata(src_path)
                    if src_path
                    else DocumentMetadata(
                        title="Unknown",
                        file_path=str(src_path) if src_path else "",
                        file_name=str(src_path.name) if src_path else "",
                        file_size=0,
                        file_type=(
                            self._get_mime_type(src_path.suffix)
                            if src_path
                            else "application/octet-stream"
                        ),
                        created_at="",
                        modified_at="",
                        processed_at="",
                        parser_name=self.parser_name,
                        parser_version=self.parser_version,
                    )
                )
                return Document(
                    document_id=doc_id,
                    metadata=meta,
                    full_text="",
                    processing_status=(
                        "failed" if conv_res.status.name == "ERROR" else "partial"
                    ),
                )

            # SUCCESS: use common builder
            docling_doc = conv_res.document
            file_path = src_path or Path(str(conv_res.input.file))
            return self._build_document_from_docling(
                docling_doc,
                file_path,
                data.get("start_time", time.time()),
            )

        # Execute post-processing (parallel with fallback)
        built_supported: list[Document]
        if parallelize_post:
            built_supported = _safe_map("Batch Post-Process", tasks, _post_process)
        else:
            built_supported = [_post_process(**t) for t in tasks]

        # Parse unsupported indices individually (reuse existing parse)
        built_unsupported: dict[int, Document] = {}
        for idx in unsupported_indices:
            p = normalized[idx]
            try:
                built_unsupported[idx] = self.parse(str(p), **options)
            except Exception:
                # minimal failed doc
                try:
                    doc_id = short_id(4)
                except Exception:
                    doc_id = self._compute_document_id(p)[:8]
                meta = self._create_base_metadata(p)
                built_unsupported[idx] = Document(
                    document_id=doc_id,
                    metadata=meta,
                    full_text="",
                    processing_status="failed",
                )

        # Merge results preserving input order
        result_docs: list[Document] = [None] * len(normalized)  # type: ignore
        # Place supported
        for local_idx, global_idx in enumerate(supported_indices):
            result_docs[global_idx] = built_supported[local_idx]
        # Place unsupported
        for global_idx, doc in built_unsupported.items():
            result_docs[global_idx] = doc

        # Replace any None (should not happen) with minimal stub
        for i, maybe_doc in enumerate(result_docs):
            if maybe_doc is None:
                p = normalized[i]
                doc_id = self._compute_document_id(p)
                meta = self._create_base_metadata(p)
                result_docs[i] = Document(
                    document_id=doc_id,
                    metadata=meta,
                    full_text="",
                    processing_status="failed",
                )

        # Save parsed results when enabled
        try:
            # Save for batch-converted (supported) inputs
            for local_idx, global_idx in enumerate(supported_indices):
                src_path = supported_paths[local_idx]
                doc = result_docs[global_idx]
                self._save_parsed_result_if_enabled(src_path, doc)

            # Save for unsupported that failed (minimal stubs)
            for global_idx in unsupported_indices:
                doc = result_docs[global_idx]
                if getattr(doc, "processing_status", "") == "failed":
                    self._save_parsed_result_if_enabled(normalized[global_idx], doc)
        except Exception:
            pass

        # Cleanup converted artifacts if enabled
        if self.cleanup_converted_files and to_cleanup:
            self._cleanup_files(to_cleanup)

        return result_docs

    # ---------- Conversion helpers ----------

    def _maybe_convert_inputs(
        self,
        inputs: list[Path],
        parallel: bool = False,
    ) -> tuple[list[Path], list[Path]]:
        """Convert any inputs supported by our conversion manager.

        Returns (converted_paths, to_cleanup), where to_cleanup contains only artifacts
        created by conversion (not re-used files), so we can safely delete them later.
        """
        # Use manager bulk API to leverage per-converter batching/parallelism
        try:
            results = self._conversion_manager.convert_all(
                inputs,
                output_dir=None,
                parallel=parallel,
            )
            out: list[Path] = []
            cleanup: list[Path] = []
            for src, res in zip(inputs, results):
                if res.ok and res.dst:
                    print(f"Converted to {res.dst.suffix}: {src} -> {res.dst}")
                    out.append(res.dst)
                    if res.backend != "reuse":
                        cleanup.append(res.dst)
                else:
                    if res.backend != "skip":
                        print(f"Conversion failed for {src}: {res.message}")
                    out.append(src)
            return out, cleanup
        except Exception as e:
            print(f"Batch conversion error: {e}")
            return inputs, []

    def _cleanup_files(self, paths: list[Path]) -> None:
        for p in paths:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                continue

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

        return self._build_document_from_docling(result.document, file_path, start_time)

    def _build_document_from_docling(
        self,
        docling_doc,
        file_path: Path,
        start_time: float,
    ) -> Document:
        """Common post-conversion pipeline used by both single and batch parsing."""
        # Create base metadata
        metadata = self._create_base_metadata(file_path)

        # Extract document statistics
        full_text = docling_doc.export_to_text()
        metadata.total_characters = len(full_text)
        metadata.total_words = len(full_text.split())

        # Extract page count from Docling document
        try:
            if hasattr(docling_doc, "pages") and docling_doc.pages:
                metadata.total_pages = len(docling_doc.pages)
        except Exception:
            pass

        # Create document id as exactly 5 characters (no hash suffix)
        try:
            document_id = short_id(4)
        except Exception:
            document_id = "docid"
        document = Document(
            document_id=document_id,
            metadata=metadata,
            full_text=full_text,
            processing_status="processing",
        )

        # Extract content structure
        # Build a Docling index for robust header/path mapping (per-document)
        try:
            doc_index = self._index_docling_structure(docling_doc)
        except Exception:
            doc_index = None

        self._extract_document_structure(docling_doc, document, doc_index=doc_index)

        # Extract images if requested
        if self.extract_images:
            self._extract_images(docling_doc, document, doc_index=doc_index)

        # Extract tables if requested
        if self.extract_tables:
            self._extract_tables(docling_doc, document, doc_index=doc_index)

        # Append image annotations and table HTML to full_text for completeness
        try:
            extras: list[str] = []
            # Image annotations
            images = getattr(document.metadata, "images", []) or []
            annotations: list[str] = []
            for img in images:
                try:
                    ann = (getattr(img, "annotation", "") or "").strip()
                    if ann:
                        annotations.append(ann)
                except Exception:
                    continue
            if annotations:
                extras.append("Image Annotations:\n" + "\n".join(annotations))

            # Tables HTML
            tables = getattr(document.metadata, "tables", []) or []
            html_blocks: list[str] = []
            for tbl in tables:
                try:
                    html = (getattr(tbl, "html", "") or "").strip()
                    if html:
                        html_blocks.append(html)
                except Exception:
                    continue
            if html_blocks:
                extras.append("Tables (HTML):\n" + "\n\n".join(html_blocks))

            if extras:
                document.full_text = (
                    (document.full_text or "") + "\n\n" + "\n\n".join(extras)
                )
        except Exception:
            pass

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
        document.metadata.processing_time = time.time() - start_time
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
        metadata.total_characters = len(content)
        metadata.total_words = len(content.split())

        # Create document id as exactly 5 characters (no hash suffix)
        try:
            document_id = short_id(4)
        except Exception:
            document_id = "docid"
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
        document.metadata.processing_time = time.time() - start_time
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

    def _extract_document_structure(
        self,
        docling_doc,
        document: Document,
        *,
        doc_index: dict | None = None,
    ):
        """Extract hierarchical document structure with waterfall fallback."""
        extraction_successful = False

        # Waterfall fallback structure
        # 1. Try hybrid chunking if enabled
        if self.use_hybrid_chunking and self.hybrid_chunker:
            try:
                self._extract_with_hybrid_chunking(
                    docling_doc,
                    document,
                    doc_index=doc_index,
                )
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
                try:
                    _para_id = short_id(4)
                except Exception:
                    _para_id = str(paragraph_id)

                paragraph = DocumentParagraph(
                    text=combined_text,
                    paragraph_id=_para_id,
                    section_id=current_section.section_id,
                    document_id=document.document_id,
                    paragraph_index=len(current_section.paragraphs),
                    metadata=paragraph_metadata.copy(),
                )

                # Split into sentences
                sentences = self._split_into_sentences(
                    combined_text,
                    _para_id,
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

                # Check if this is a heading based on Docling label first (airtight)
                is_heading = False
                docling_label = (
                    str(getattr(item, "label", "")).lower()
                    if hasattr(item, "label")
                    else ""
                )
                if (
                    "section_header" in docling_label
                    or docling_label.endswith("title")
                    or docling_label == "title"
                ):
                    is_heading = True
                elif hasattr(item, "text") and item.text:
                    # Heuristic fallback only if Docling didn't label it
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
                    # Prefer Docling's explicit heading level when available
                    header_level = getattr(item, "level", None)
                    if not isinstance(header_level, int):
                        header_level = level if level is not None else 1

                    # Pop sections from stack for proper hierarchy
                    while section_stack and section_stack[-1][0] >= header_level:
                        section_stack.pop()

                    # Create new section
                    # Compute path from existing stack + this header
                    base_path = [s.title for (_lvl, s) in section_stack] + [title_text]
                    current_section = DocumentSection(
                        title=title_text,
                        section_id=short_id(5),
                        document_id=document.document_id,
                        section_index=section_id - 1,
                        metadata={
                            "level": header_level,
                            "docling_type": item_type,
                            "docling_label": (
                                str(item.label) if hasattr(item, "label") else None
                            ),
                            "path": base_path,
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
                            section_id=short_id(5),
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
                                            prov_item.bbox.model_dump(
                                                exclude=["coord_origin"],
                                            )
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
                            section_id=short_id(5),
                            document_id=document.document_id,
                            section_index=0,
                        )
                        document.sections.append(current_section)
                        section_stack.append((0, current_section))
                        section_id += 1

                    # Create paragraph with list metadata
                    try:
                        _para_id = short_id(4)
                    except Exception:
                        _para_id = str(paragraph_id)

                    paragraph = DocumentParagraph(
                        text=item.text.strip(),
                        paragraph_id=_para_id,
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
                    try:
                        _sent_id = short_id(4)
                    except Exception:
                        _sent_id = str(sentence_id)
                    paragraph.sentences = [
                        DocumentSentence(
                            text=item.text.strip(),
                            sentence_id=_sent_id,
                            paragraph_id=_para_id,
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
                            section_id=short_id(5),
                            document_id=document.document_id,
                            section_index=0,
                        )
                        document.sections.append(current_section)
                        section_stack.append((0, current_section))
                        section_id += 1

                    # Create specialized paragraph
                    try:
                        _para_id2 = short_id(4)
                    except Exception:
                        _para_id2 = str(paragraph_id)

                    paragraph = DocumentParagraph(
                        text=item.text.strip(),
                        paragraph_id=_para_id2,
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
                    try:
                        _sent_id2 = short_id(4)
                    except Exception:
                        _sent_id2 = str(sentence_id)
                    paragraph.sentences = [
                        DocumentSentence(
                            text=item.text.strip(),
                            sentence_id=_sent_id2,
                            paragraph_id=_para_id2,
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
                        section_id=short_id(5),
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
                            section_id=short_id(5),
                            document_id=document.document_id,
                            section_index=0,
                        )
                        document.sections.append(current_section)
                        section_id += 1

                    # Create paragraph
                    try:
                        _para_id = short_id(4)
                    except Exception:
                        _para_id = str(paragraph_id)

                    paragraph = DocumentParagraph(
                        text=text_item.text.strip(),
                        paragraph_id=_para_id,
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
                        _para_id,
                        current_section.section_id,
                        document.document_id,
                        sentence_id,
                    )
                    paragraph.sentences = sentences
                    sentence_id += len(sentences)

                    current_section.paragraphs.append(paragraph)
                    paragraph_id += 1

    def _extract_with_hybrid_chunking(
        self,
        docling_doc,
        document: Document,
        *,
        doc_index: dict | None = None,
    ):
        """Extract structure using Docling's hybrid chunker with robust hierarchy building."""
        # 1) Get chunks from the hybrid chunker
        chunks = list(self.hybrid_chunker.chunk(docling_doc))

        # 2) Pre-compute enriched context for each chunk (used for embeddings and heuristics)
        chunk_text_map: dict[str, str] = {}
        for c in chunks:
            try:
                key = (c.text or "").strip()
                if key:
                    chunk_text_map[key] = self.enrich_chunk_context(c)
            except Exception:
                pass

        # 3) Build a robust hierarchical map using headings path and doc items
        #    Note: enrichment is done above before building hierarchy
        chunk_hierarchy = self._build_chunk_hierarchy(
            chunks,
            doc_index=doc_index,
            chunk_text_map=chunk_text_map,
        )

        # 4) Iterate and construct sections/paragraphs
        section_id = 1
        paragraph_id = 1
        sentence_id = 1

        # Map from heading path tuple to created DocumentSection
        created_sections: dict[tuple[str, ...], DocumentSection] = {}
        current_section: DocumentSection | None = None

        for i, chunk in enumerate(chunks):
            raw_text = (chunk.text or "").strip()
            if not raw_text:
                continue

            ctx = chunk_hierarchy.get(i, {})
            headings: list[str] = list(ctx.get("headings") or [])
            is_heading: bool = bool(ctx.get("is_heading", False))
            # Prefer explicit level, otherwise derive from headings length
            heading_level: int = int(
                ctx.get("level") or (len(headings) if headings else 0),
            )

            # Ensure sections exist for the full heading path
            if headings:
                for depth in range(1, len(headings) + 1):
                    key = tuple(headings[:depth])
                    if key not in created_sections:
                        new_section = DocumentSection(
                            title=headings[depth - 1],
                            section_id=short_id(5),
                            document_id=document.document_id,
                            section_index=section_id - 1,
                            level=depth,
                            metadata={
                                "level": depth,
                                "path": list(headings[:depth]),
                                "from_chunk_headings": True,
                            },
                        )
                        document.sections.append(new_section)
                        created_sections[key] = new_section
                        current_section = new_section
                        section_id += 1

                current_section = created_sections.get(tuple(headings))

            elif is_heading:
                # No meta.headings, but strong signal of header in text/doc_items
                # Build a path by appending to current path when available
                base_path: list[str] = []
                try:
                    if current_section and isinstance(current_section.metadata, dict):
                        base_path = list(current_section.metadata.get("path") or [])
                except Exception:
                    base_path = []

                path = base_path + [raw_text]
                for depth in range(1, len(path) + 1):
                    key = tuple(path[:depth])
                    if key not in created_sections:
                        new_section = DocumentSection(
                            title=path[depth - 1],
                            section_id=short_id(5),
                            document_id=document.document_id,
                            section_index=section_id - 1,
                            level=depth,
                            metadata={
                                "level": depth,
                                "path": list(path[:depth]),
                                "from_detected_heading": True,
                            },
                        )
                        document.sections.append(new_section)
                        created_sections[key] = new_section
                        current_section = new_section
                section_id += 1

                current_section = created_sections.get(tuple(path))

            # Fallback: ensure a default section exists
            if current_section is None:
                key = ("Document Content",)
                if key not in created_sections:
                    default_section = DocumentSection(
                        title="Document Content",
                        section_id=short_id(5),
                        document_id=document.document_id,
                        section_index=0,
                        level=1,
                        metadata={"level": 1, "path": ["Document Content"]},
                    )
                    document.sections.append(default_section)
                    created_sections[key] = default_section
                    section_id += 1
                current_section = created_sections[key]

            # Skip adding a paragraph if this chunk is a pure heading line
            anchor_heading = headings[-1] if headings else None
            if (
                is_heading
                and anchor_heading
                and raw_text.strip() == anchor_heading.strip()
            ):
                # pure header chunk, do not duplicate as paragraph
                continue

            # Validate chunk token count (contextualized)
            _ = self.validate_chunk_tokens(chunk)

            # Enriched context string (for embeddings, overlap)
            try:
                enriched_text = chunk_text_map.get(
                    raw_text,
                ) or self.enrich_chunk_context(chunk)
            except Exception:
                enriched_text = raw_text

                # Create paragraph
                paragraph = DocumentParagraph(
                    text=raw_text,
                    paragraph_id=str(paragraph_id),
                    section_id=current_section.section_id,
                    document_id=document.document_id,
                    paragraph_index=len(current_section.paragraphs),
                )

                # Assemble rich metadata for the paragraph
                chunk_metadata = self.get_chunk_metadata(chunk)
                paragraph.metadata.update(
                    {
                        "chunk_index": i,
                        "chunk_valid_tokens": True,
                        "enriched_context": enriched_text,
                        "enriched_length": len(enriched_text or ""),
                        "headings_path": headings,
                        "detected_heading": bool(is_heading),
                        "detected_level": heading_level,
                    },
                )
            paragraph.metadata.update(chunk_metadata)

            # Provenance: pages and bboxes from doc_items
            if hasattr(chunk, "meta") and getattr(chunk.meta, "doc_items", None):
                pages: set[int] = set()
                bboxes: list[dict] = []
                labels: list[str] = []
                for doc_item in chunk.meta.doc_items:
                    try:
                        if hasattr(doc_item, "label") and doc_item.label:
                            labels.append(str(doc_item.label))
                        if hasattr(doc_item, "prov") and doc_item.prov:
                            for prov in doc_item.prov:
                                if (
                                    hasattr(prov, "page_no")
                                    and prov.page_no is not None
                                ):
                                    pages.add(prov.page_no)
                                if hasattr(prov, "bbox") and getattr(
                                    prov,
                                    "bbox",
                                    None,
                                ):
                                    bboxes.append(
                                        {
                                            "l": getattr(prov.bbox, "l", 0),
                                            "t": getattr(prov.bbox, "t", 0),
                                            "r": getattr(prov.bbox, "r", 0),
                                            "b": getattr(prov.bbox, "b", 0),
                                        },
                                    )
                    except Exception:
                        continue
                if pages:
                    paragraph.metadata["pages"] = sorted(pages)
                if bboxes:
                    paragraph.metadata["bboxes"] = bboxes
                if labels:
                    paragraph.metadata["doc_item_labels"] = labels

            # Split into sentences with contextual awareness
            sentences = self._split_into_sentences_contextual(
                raw_text,
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

    def _build_chunk_hierarchy(
        self,
        chunks,
        *,
        doc_index: dict | None = None,
        chunk_text_map: dict | None = None,
    ) -> Dict[int, Dict]:
        """Build hierarchical context for chunks using DocMeta.headings and strong heuristics.

        Returns a dictionary keyed by chunk index with fields:
        - headings: full headings path (list[str])
        - is_heading: whether this chunk is likely a heading line
        - level: detected heading level (int)
        - parent_heading: immediate parent heading text or None
        - doc_items: original doc items (for reference)
        - pages: list of page numbers if available
        - section_path: tuple of headings (stable key)
        - anchor_heading: last heading in path if any
        - start_new_section: True if this chunk starts a new section
        """
        hierarchy: dict[int, dict] = {}
        prev_headings: list[str] = []

        import re

        for i, chunk in enumerate(chunks):
            text = (getattr(chunk, "text", "") or "").strip()
            ctx: dict[str, Any] = {
                "is_heading": False,
                "level": 0,
                "parent_heading": None,
                "doc_items": [],
                "headings": [],
            }

            # Pull headings path from metadata when present
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    try:
                        ctx["headings"] = [str(h) for h in chunk.meta.headings]
                    except Exception:
                        ctx["headings"] = list(chunk.meta.headings)

                if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    ctx["doc_items"] = chunk.meta.doc_items

            headings: list[str] = ctx["headings"]

            # 0) Strengthen headings using Docling index when doc_items are available
            try:
                if doc_index and ctx["doc_items"]:
                    # Collect candidate paths for all doc_items in this chunk
                    paths: list[tuple[str, ...]] = []
                    for di in ctx["doc_items"]:
                        ref = getattr(di, "self_ref", None)
                        if ref and ref in doc_index.get("ref_to_path", {}):
                            paths.append(doc_index["ref_to_path"][ref])
                    if paths:
                        # Choose most frequent path
                        from collections import Counter

                        common_path, _count = Counter(paths).most_common(1)[0]
                        headings = list(common_path)
                        ctx["headings"] = headings
            except Exception:
                pass

            # Determine if this chunk itself is a heading
            # 1) Doc item labels / levels
            level_from_item: int | None = None
            for di in ctx["doc_items"]:
                try:
                    if hasattr(di, "label") and di.label:
                        lab = str(di.label).lower()
                        if lab in (
                            "title",
                            "heading",
                            "section_header",
                        ) or lab.startswith("h"):
                            ctx["is_heading"] = True
                            if hasattr(di, "level") and di.level is not None:
                                level_from_item = int(di.level)
                            elif (
                                lab.startswith("h")
                                and len(lab) > 1
                                and lab[1].isdigit()
                            ):
                                level_from_item = int(lab[1])
                            elif lab == "title":
                                level_from_item = 1
                except Exception:
                    continue

            # 2) Text heuristics
            if not ctx["is_heading"] and text and len(text) < 200:
                if re.match(r"^[\d]+(\.[\d]+)*\s+\S+", text):
                    ctx["is_heading"] = True
                elif text.isupper() and len(text.split()) < 12:
                    ctx["is_heading"] = True
                elif re.match(
                    r"^(Chapter|Section|Part|Appendix)\s+[\dIVXLCDM]+",
                    text,
                    re.I,
                ):
                    ctx["is_heading"] = True
                elif self._is_likely_header(text, chunk):
                    ctx["is_heading"] = True

            # 2b) If any doc_item is exactly a Docling heading (by ref), prefer that
            try:
                if doc_index and ctx["doc_items"] and not ctx["is_heading"]:
                    for di in ctx["doc_items"]:
                        ref = getattr(di, "self_ref", None)
                        if ref and ref in doc_index.get("heading_refs", set()):
                            ctx["is_heading"] = True
                            # Prefer explicit level if available from Docling
                            if level_from_item is None:
                                level_from_item = doc_index.get(
                                    "heading_ref_to_level",
                                    {},
                                ).get(ref, None)
                            break
            except Exception:
                pass

            # 3) If enriched contextual text starts with the last heading, mark as header
            if headings:
                try:
                    enriched = (chunk_text_map or {}).get(
                        text,
                    ) or self.enrich_chunk_context(chunk)
                    first_line = (enriched or "").split("\n", 1)[0].strip()
                    if first_line and first_line.strip() == headings[-1].strip():
                        ctx["is_heading"] = True
                except Exception:
                    pass

            # Determine level preference: doc item > headings path length > heuristic
            if level_from_item is not None:
                ctx["level"] = int(level_from_item)
            elif headings:
                ctx["level"] = len(headings)
            elif ctx["is_heading"]:
                ctx["level"] = max(1, self._estimate_heading_level(text))
            else:
                ctx["level"] = 0

            # Parent and keys
            if len(headings) >= 2:
                ctx["parent_heading"] = headings[-2]
            ctx["section_path"] = tuple(headings)
            ctx["anchor_heading"] = headings[-1] if headings else None

            # Provenance pages
            pages: set[int] = set()
            try:
                for di in ctx["doc_items"]:
                    if hasattr(di, "prov") and di.prov:
                        for prov in di.prov:
                            if hasattr(prov, "page_no") and prov.page_no is not None:
                                pages.add(prov.page_no)
                if pages:
                    ctx["pages"] = sorted(pages)
            except Exception:
                pass

            # Start new section if headings path changes or we detect a header
            ctx["heading_changed"] = headings != prev_headings
            ctx["start_new_section"] = bool(ctx["heading_changed"] or ctx["is_heading"])

            hierarchy[i] = ctx
            prev_headings = headings

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

    # Note: avoid per-document caches on self to stay concurrency-safe

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
                section_id=short_id(4),
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

                try:
                    _para_id = short_id(4)
                except Exception:
                    _para_id = str(paragraph_id)

                paragraph = DocumentParagraph(
                    text=chunk_text.strip(),
                    paragraph_id=_para_id,
                    section_id=section.section_id,
                    document_id=document.document_id,
                    paragraph_index=len(section.paragraphs),
                )

                # Split paragraph into sentences
                sentences = self._split_into_sentences(
                    chunk_text.strip(),
                    _para_id,
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
        """Paragraph splitting with fallbacks: LangChain > regex."""
        try:
            src = (text or "").strip()
            if not src:
                return []
            # Prefer LangChain paragraph splitter
            if LANGCHAIN_AVAILABLE and self.paragraph_splitter is not None:
                parts = [
                    p
                    for p in self.paragraph_splitter.split_text(src)
                    if p and p.strip()
                ]
                return parts
        except Exception:
            pass
        # Regex fallback
        chunks = src.split("\n\n")
        if len(chunks) <= 2 and len(src) > 1000:
            import re

            alt_chunks = re.split(r"\n(?=[A-Z])", src)
            if len(alt_chunks) > len(chunks) * 2:
                chunks = alt_chunks
        paragraphs = []
        for chunk in chunks:
            cleaned = chunk.strip()
            if cleaned and len(cleaned) > 20:
                paragraphs.append(cleaned)
        return paragraphs

    def _split_into_sentences(
        self,
        text: str,
        paragraph_id: Union[int, str],
        section_id: Union[int, str],
        document_id: Union[int, str],
        start_sentence_id: int,
    ) -> List[DocumentSentence]:
        """Split text into sentences with spaCy > LangChain > regex fallbacks."""
        import re

        text_norm = re.sub(r"\s+", " ", (text or "").strip())
        if not text_norm:
            return []
        # 1) spaCy (best for sentence boundaries, handles abbreviations/punct)
        if SPACY_AVAILABLE and self._spacy_nlp is not None:
            try:
                doc = self._spacy_nlp(text_norm)
                chunks = [
                    s.text.strip() for s in doc.sents if s.text and s.text.strip()
                ]
            except Exception:
                chunks = []
        else:
            chunks = []
        # 2) LangChain splitter
        if not chunks and LANGCHAIN_AVAILABLE and self.sentence_splitter is not None:
            try:
                chunks = [
                    c.strip()
                    for c in self.sentence_splitter.split_text(text_norm)
                    if c and c.strip()
                ]
            except Exception:
                chunks = []
        # 3) Regex fallback
        if not chunks:
            chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"])", text_norm)
        # Clean leading punctuation defensively
        cleaned_chunks: List[str] = []
        for s in chunks:
            s2 = re.sub(r"^[\s\-–—•·]*([.!?])+\s*", "", s).strip()
            if s2:
                cleaned_chunks.append(s2)
        sentences: List[DocumentSentence] = []
        for i, s in enumerate(cleaned_chunks):
            try:
                _sent_id = short_id(5)
            except Exception:
                _sent_id = str(start_sentence_id + i)
            sentences.append(
                DocumentSentence(
                    text=s,
                    sentence_id=_sent_id,
                    paragraph_id=str(paragraph_id),
                    section_id=str(section_id),
                    document_id=str(document_id),
                    sentence_index=i,
                ),
            )
        return sentences

    def _extract_images(
        self,
        docling_doc,
        document: Document,
        *,
        doc_index: dict | None = None,
    ):
        """Extract images from the document."""
        try:
            images: list[DocumentImage] = []

            # Look for pictures in page elements
            if hasattr(docling_doc, "pictures"):
                for picture in docling_doc.pictures:
                    if (
                        hasattr(picture, "label")
                        and "picture" in str(picture.label).lower()
                    ):
                        try:
                            # Get the provenance items from the picture
                            page_num, bbox, annotation_text, annotation_provenance = (
                                None,
                                None,
                                None,
                                None,
                            )
                            provenance_items = picture.prov
                            annotations = picture.annotations

                            for provenance_item in provenance_items:
                                page_num = provenance_item.page_no
                                bbox = provenance_item.bbox
                                break

                            for annotation in annotations:
                                if not isinstance(annotation, PictureDescriptionData):
                                    continue
                                annotation_text = annotation.text
                                annotation_provenance = annotation.provenance
                                break

                            # Try to infer section path from the picture node's self_ref
                            section_path = None
                            try:
                                if doc_index is not None and hasattr(
                                    picture,
                                    "self_ref",
                                ):
                                    ref = getattr(picture, "self_ref", None)
                                    if ref and ref in doc_index.get("ref_to_path", {}):
                                        section_path = list(
                                            doc_index["ref_to_path"][ref],
                                        )
                            except Exception:
                                section_path = None

                            images.append(
                                DocumentImage(
                                    page=page_num,
                                    bbox=(
                                        bbox.model_dump(exclude=["coord_origin"])
                                        if bbox
                                        else {}
                                    ),
                                    element_type=str(picture.label),
                                    annotation=annotation_text,
                                    annotation_provenance=annotation_provenance,
                                    section_path=section_path,
                                ),
                            )
                        except Exception:
                            pass

            document.metadata.images = images

        except Exception:
            pass

    def _extract_tables(
        self,
        docling_doc,
        document: Document,
        *,
        doc_index: dict | None = None,
    ):
        """Extract table data from the document."""
        try:
            tables: list[DocumentTable] = []

            if hasattr(docling_doc, "tables"):
                for table in docling_doc.tables:
                    if hasattr(table, "label") and "table" in str(table.label).lower():
                        try:
                            # Get the provenance items from the table
                            page_num, bbox = None, None
                            provenance_items = table.prov

                            for provenance_item in provenance_items:
                                page_num = provenance_item.page_no
                                bbox = provenance_item.bbox
                                break

                            # Try to infer section path from table self_ref
                            section_path = None
                            try:
                                if doc_index is not None and hasattr(table, "self_ref"):
                                    ref = getattr(table, "self_ref", None)
                                    if ref and ref in doc_index.get("ref_to_path", {}):
                                        section_path = list(
                                            doc_index["ref_to_path"][ref],
                                        )
                            except Exception:
                                section_path = None

                            tables.append(
                                DocumentTable(
                                    page=page_num,
                                    element_type=str(table.label),
                                    html=table.export_to_html(doc=docling_doc),
                                    bbox=(
                                        bbox.model_dump(exclude=["coord_origin"])
                                        if bbox
                                        else {}
                                    ),
                                    section_path=section_path,
                                ),
                            )
                        except Exception:
                            pass

            document.metadata.tables = tables

        except Exception:
            pass

    def _extract_enhanced_metadata(self, document: Document):
        """
        Extract **structured, retrieval-ready** metadata via a token-aware LLM flow.

        Overview
        --------
        Calls an LLM (o4-mini) with a strict, Pydantic-validated JSON schema to
        produce metadata suitable for downstream retrieval (classification, topics,
        entities, tags, confidence). The routine is **token-aware** and clips or
        samples text to respect the summariser's context budget while maximising
        usable signal.

        Token Budgets (env-configurable)
        --------------------------------
        SUMMARY_ENCODING
            tiktoken encoding used for summariser accounting (default: "o200k_base").
        SUMMARY_MAX_TOKENS
            Upper bound for prompt+text per call (default: 100000).
        SUMMARY_REDUCE_STEP
            Backoff step when token errors occur (default: 5000).
        SUMMARY_MIN_TOKENS
            Minimum floor during backoff (default: 4000).

        Source Selection (priority)
        ---------------------------
        1) Full document text (token-aware clipped); if it exceeds budget, also
        try a **head/middle/tail** token-sampled variant.
        2) Document summary, optionally augmented with a token-aware subset of
        section summaries (no fixed counts).
        3) A token-aware concatenation of available section summaries (no fixed counts).
        4) A token-aware sample of paragraph text/summary from early sections (no fixed counts).
        5) Raw `document.content` fallback, if present.

        Behaviour
        ---------
        • Each candidate source is clipped to the current **prompt-aware** budget.
        • On token errors, the routine retries with a reduced budget (backoff) up to
        a small limit before moving on to the next source.
        • On success, the JSON is validated against `DocumentMetadataExtraction` and
        `document.metadata` is populated in place.
        • If all sources fail, the function returns without mutating metadata.
        """
        try:
            import unify

            client = unify.Unify(
                "o4-mini@openai",
                cache=json.loads(os.environ.get("UNIFY_CACHE", "true")),
            )

            # Token accounting (align with main summariser defaults; no fixed caps)
            METADATA_ENCODING = os.environ.get("SUMMARY_ENCODING", "o200k_base")
            try:
                METADATA_MAX_TOKENS = int(
                    os.environ.get("SUMMARY_MAX_TOKENS", "100000"),
                )
            except Exception:
                METADATA_MAX_TOKENS = 100000
            try:
                METADATA_REDUCE_STEP = int(
                    os.environ.get("SUMMARY_REDUCE_STEP", "5000"),
                )
            except Exception:
                METADATA_REDUCE_STEP = 5000
            try:
                METADATA_MIN_TOKENS = int(os.environ.get("SUMMARY_MIN_TOKENS", "4000"))
            except Exception:
                METADATA_MIN_TOKENS = 4000

            MAX_RETRY_ATTEMPTS = 3

            # Prompt and prompt-aware budget
            from unity.file_manager.parser.prompt_builders import (
                build_metadata_extraction_prompt,
            )

            prompt = build_metadata_extraction_prompt()
            prompt_tokens = conservative_token_estimate(prompt, METADATA_ENCODING)

            def _budget_for_text(current_total_budget: int) -> int:
                """Return usable token budget for the **text** after accounting for prompt."""
                return max(current_total_budget - prompt_tokens, 256)

            # ---------- token helpers (token-accurate slicing; char fallback) ----------
            def _first_tokens(text: str, n: int, enc: str) -> str:
                try:
                    import tiktoken

                    e = tiktoken.get_encoding(enc)
                    toks = e.encode(text)
                    return e.decode(toks[: max(n, 0)])
                except Exception:
                    return text[: n * 4]

            def _last_tokens(text: str, n: int, enc: str) -> str:
                try:
                    import tiktoken

                    e = tiktoken.get_encoding(enc)
                    toks = e.encode(text)
                    return e.decode(toks[-max(n, 0) :]) if n > 0 else ""
                except Exception:
                    return text[-(n * 4) :] if n > 0 else ""

            def _middle_tokens(text: str, n: int, enc: str) -> str:
                try:
                    import tiktoken

                    e = tiktoken.get_encoding(enc)
                    toks = e.encode(text)
                    L = len(toks)
                    if L == 0 or n <= 0:
                        return ""
                    start = max((L // 2) - (n // 2), 0)
                    end = min(start + n, L)
                    return e.decode(toks[start:end])
                except Exception:
                    approx = n * 4
                    s = max((len(text) // 2) - (approx // 2), 0)
                    return text[s : s + approx]

            # ---------- composition helpers (no fixed counts) --------------------------
            def _append_token_safe(
                base: str,
                addition: str,
                enc: str,
                limit: int,
            ) -> tuple[str, bool]:
                """
                Append `addition` to `base` if it fits within `limit` tokens (encoding `enc`).
                If it would exceed the limit, try a clipped version of `addition`.
                Returns (new_text, did_append_anything).
                """
                if not has_meaningful_text(addition):
                    return base, False
                # Fast path: try full addition
                combined = (
                    (base + ("\n\n" if base else "") + addition) if base else addition
                )
                if count_tokens_per_utf_byte(combined) <= limit:
                    return combined, True
                # Try clipped addition
                remaining = max(limit - count_tokens_per_utf_byte(base), 0)
                if remaining <= 0:
                    return base, False
                clipped = clip_text_to_token_limit_conservative(
                    addition,
                    remaining,
                    enc,
                )
                if has_meaningful_text(clipped):
                    combined = (
                        (base + ("\n\n" if base else "") + clipped) if base else clipped
                    )
                    return combined, True
                return base, False

            def _gather_section_summaries(enc: str, limit: int) -> str:
                """
                Accumulate section summaries until the token limit is reached.
                No fixed number of sections is assumed.
                """
                if not getattr(document, "sections", None):
                    return ""
                out = ""
                for idx, section in enumerate(document.sections, 1):
                    block = ""
                    title = (getattr(section, "title", "") or "").strip()
                    if has_meaningful_text(title):
                        block = f"Section {idx}: {title}"
                    summ = (getattr(section, "summary", "") or "").strip()
                    if has_meaningful_text(summ):
                        block = block + ("\n" if block else "") + summ
                    if not has_meaningful_text(block):
                        continue
                    new_out, appended = _append_token_safe(out, block, enc, limit)
                    if not appended:
                        break
                    out = new_out
                return out

            def _gather_paragraph_samples(enc: str, limit: int) -> str:
                """
                Accumulate paragraph summaries/text across early sections in order,
                stopping when token limit is reached. No fixed counts.
                """
                if not getattr(document, "sections", None):
                    return ""
                out = ""
                for s_idx, section in enumerate(document.sections, 1):
                    title = (getattr(section, "title", "") or "").strip()
                    header = (
                        f"Section {s_idx}: {title}"
                        if has_meaningful_text(title)
                        else f"Section {s_idx}"
                    )
                    block, appended = _append_token_safe(out, header, enc, limit)
                    if appended:
                        out = block
                    if not getattr(section, "paragraphs", None):
                        continue
                    for para in section.paragraphs:
                        src = (
                            getattr(para, "summary", "")
                            or getattr(para, "text", "")
                            or ""
                        ).strip()
                        if not has_meaningful_text(src):
                            continue
                        out2, appended = _append_token_safe(out, src, enc, limit)
                        if not appended:
                            return out
                        out = out2
                return out

            # ---------- safe LLM call with backoff -------------------------------------
            def _safe_generate_json(
                prompt_str: str,
                text: str,
                total_budget: int,
            ) -> str:
                usable = _budget_for_text(total_budget)
                clipped = clip_text_to_token_limit_conservative(
                    text,
                    usable,
                    METADATA_ENCODING,
                )
                return client.copy().generate(prompt_str + clipped)

            # ---------- Build candidate sources (in priority order) ---------------------
            text_sources: list[tuple[str, str] | tuple[str, callable]] = []

            # 1) Full document text (+ token-sampled variant when oversized)
            full_text = getattr(document, "full_text", "") or ""
            if has_meaningful_text(full_text):
                text_sources.append(("full_text", full_text))
                if not is_within_token_limit_conservative(
                    full_text,
                    _budget_for_text(METADATA_MAX_TOKENS),
                    METADATA_ENCODING,
                ):
                    # split budget across head/middle/tail portions
                    each = max(_budget_for_text(METADATA_MAX_TOKENS) // 3, 128)
                    sampled = (
                        f"{first_tokens_per_utf_byte(full_text, each)}\n\n"
                        "[...Document middle section...]\n\n"
                        f"{middle_tokens_per_utf_byte(full_text, each)}\n\n"
                        "[...Document end section...]\n\n"
                        f"{last_tokens_per_utf_byte(full_text, each)}"
                    )
                    text_sources.append(("full_text_sampled", sampled))

            # 2) Document summary optionally augmented by a token-aware subset of section summaries
            doc_summary = (getattr(document, "summary", "") or "").strip()
            if has_meaningful_text(doc_summary):

                def _doc_summary_augmented() -> str:
                    # Reserve 3/4 for summary, 1/4 for extras (no constants exposed; ratio-based)
                    total = _budget_for_text(METADATA_MAX_TOKENS)
                    main_budget = max(int(total * 0.75), 256)
                    aux_budget = max(total - main_budget, 128)
                    main = clip_text_to_token_limit_conservative(
                        doc_summary,
                        main_budget,
                        METADATA_ENCODING,
                    )
                    aux = _gather_section_summaries(METADATA_ENCODING, aux_budget)
                    return main if not has_meaningful_text(aux) else f"{main}\n\n{aux}"

                text_sources.append(("document_summary_aug", _doc_summary_augmented))

            # 3) Token-aware concatenation of available section summaries
            def _all_section_summaries() -> str:
                return _gather_section_summaries(
                    METADATA_ENCODING,
                    _budget_for_text(METADATA_MAX_TOKENS),
                )

            text_sources.append(("section_summaries", _all_section_summaries))

            # 4) Token-aware paragraph sampling
            def _paragraph_samples() -> str:
                return _gather_paragraph_samples(
                    METADATA_ENCODING,
                    _budget_for_text(METADATA_MAX_TOKENS),
                )

            text_sources.append(("paragraph_samples", _paragraph_samples))

            # 5) Raw content fallback
            raw_content = (getattr(document, "content", "") or "").strip()
            if has_meaningful_text(raw_content):
                text_sources.append(("raw_content", raw_content))

            # ---------- Try sources with progressive backoff ----------------------------
            current_budget = METADATA_MAX_TOKENS
            last_error: Exception | None = None

            for name, src in text_sources:
                # Resolve callables lazily to honour current budget
                src_text = src() if callable(src) else src
                if not has_meaningful_text(src_text):
                    continue

                for attempt in range(MAX_RETRY_ATTEMPTS):
                    try:
                        response = _safe_generate_json(prompt, src_text, current_budget)
                        validated = DocumentMetadataExtraction.model_validate_json(
                            response,
                        )

                        # Populate metadata (in place)
                        document.metadata.document_type = validated.document_type
                        document.metadata.category = validated.category
                        document.metadata.key_topics = validated.key_topics
                        document.metadata.named_entities = validated.named_entities
                        document.metadata.content_tags = validated.content_tags
                        document.metadata.confidence_score = validated.confidence_score

                        # Defensive normalisation of large lists (metadata, not embeddings)
                        if isinstance(document.metadata.key_topics, list):
                            document.metadata.key_topics = document.metadata.key_topics[
                                :256
                            ]
                        if isinstance(document.metadata.content_tags, list):
                            document.metadata.content_tags = (
                                document.metadata.content_tags[:256]
                            )

                        return  # success

                    except Exception as err:
                        last_error = err
                        # Token-related backoff; then retry
                        if (
                            "token" in str(err).lower()
                            and current_budget > METADATA_MIN_TOKENS
                            and attempt < MAX_RETRY_ATTEMPTS - 1
                        ):
                            current_budget = max(
                                current_budget - METADATA_REDUCE_STEP,
                                METADATA_MIN_TOKENS,
                            )
                            continue
                        # Otherwise, break and try next source
                        break

            # If all sources fail, leave metadata unchanged (caller may fallback or ignore)
            if last_error:
                pass

        except Exception:
            # Silent failure – do not mutate metadata on unexpected errors
            pass

    def _generate_summaries(self, document: Document):
        """
        Generate **hierarchical, token-aware** summaries (paragraph → section → document)
        using an LLM pipeline with parallel map-reduce and robust fallbacks.

        Overview
        --------
        This routine orchestrates a three-tier summarisation pipeline:
        1) Paragraph summaries (fine-grained, parallel)
        2) Section summaries synthesised from paragraph summaries (parallel)
        3) Document summary synthesised from section summaries (single pass)

        All steps are **token-aware** and will chunk/clip inputs so they respect both:
        • The long-context summariser model limits (o4-mini via `SUMMARY_*`)
        • The downstream embedding model limits (`EMBEDDING_*`) — final outputs are
            clipped to ensure they are embeddable without additional processing.

        Environment Variables (tunable)
        --------------------------------
        SUMMARY_ENCODING
            tiktoken encoding used for the **summariser** model context accounting.
            Default: "o200k_base" (compatible with o4/gpt-4o/gpt-4o-mini).

        SUMMARY_MAX_TOKENS
            Upper bound on total tokens (prompt + text) sent to the summariser
            per call before chunking begins. Large by default to leverage o4-mini.
            Default (int): 100000

        SUMMARY_REDUCE_STEP
            When a call exceeds the token budget (e.g., model/tool error), the routine
            retries with `max_tokens - SUMMARY_REDUCE_STEP`. Repeat until success or
            `SUMMARY_MIN_TOKENS` is reached. Default (int): 5000

        SUMMARY_MIN_TOKENS
            Lower bound for retry reductions; below this we fall back to token-aware
            chunking or safe clipping. Default (int): 4000

        EMBEDDING_ENCODING
            tiktoken encoding used by the **embedding** model (e.g., text-embedding-3-*).
            Default: "cl100k_base".

        EMBEDDING_MAX_INPUT_TOKENS
            Hard cap for any single summary (paragraph, section, or document) so it
            can be embedded directly without reprocessing. A small safety margin
            under the model's published limit is recommended. Default (int): 8000.

        Key Properties
        ---------------
        • Token-aware chunking:
            Long inputs are split on sentence/fragment boundaries. Adjacent chunks
            include a small **overlap** (20% by default) to preserve context continuity.
        • Progressive back-off:
            If a single-shot request exceeds the budget or the model rejects it with a
            token-related error, the routine progressively reduces the token budget
            and retries, before switching to chunked map-reduce.
        • Parallel execution (with sequential fallback):
            Paragraph and section stages use `unify.map` for concurrency; if the
            pool is unavailable, execution falls back to a sequential loop.
        • Embedding-safe outputs:
            Every paragraph, section, and the final document summary is **clipped**
            to `EMBEDDING_MAX_INPUT_TOKENS` using `clip_text_to_token_limit(...)`.
        • Robust error handling:
            Any exception in the pipeline triggers a structured fallback to
            `_generate_basic_summaries(...)`, which is also token-aware.

        Inputs
        ------
        document : Document
            A parsed document with hierarchical structure (sections → paragraphs).
            The function reads from:
            • paragraph.text
            • section.title / section.paragraphs[*].summary
            • document.full_text, document.metadata (for context)
            and writes to:
            • paragraph.summary
            • section.summary
            • document.summary

        Side Effects
        ------------
        Mutates `document` in place by populating `summary` fields at all levels.

        Returns
        -------
        None
            Results are attached to the provided `document`.

        Notes
        -----
        • Prompts used:
            - `build_paragraph_summary_prompt()`
            - `build_section_summary_prompt()`
            - `build_document_summary_prompt()`
            - `build_chunked_text_summary_prompt(chunk_number, total_chunks)`
        • Overlap ratio for chunking is fixed at 0.2 for stable behaviour; you may
        expose it via env/config if needed.
        • The routine aggressively normalises/filters empty/whitespace inputs to
        avoid generating degenerate summaries.
        """

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

            # ---------- Token accounting (tiktoken-backed) ----------
            # Model/encoding guidance:
            # - o4 / gpt-4o / gpt-4o-mini → o200k_base
            # - text-embedding-3-* → cl100k_base (used elsewhere for embeddings)
            SUMMARISER_MODEL_ENCODING = os.environ.get("SUMMARY_ENCODING", "o200k_base")
            # Use large default budgets to leverage o4-mini's long context, while remaining configurable.
            try:
                INITIAL_MAX_TOKENS = int(os.environ.get("SUMMARY_MAX_TOKENS", "100000"))
            except Exception:
                INITIAL_MAX_TOKENS = 100000
            try:
                TOKEN_REDUCTION_STEP = int(
                    os.environ.get("SUMMARY_REDUCE_STEP", "5000"),
                )
            except Exception:
                TOKEN_REDUCTION_STEP = 5000
            try:
                MIN_TOKEN_LIMIT = int(os.environ.get("SUMMARY_MIN_TOKENS", "4000"))
            except Exception:
                MIN_TOKEN_LIMIT = 4000

            # Embedding constraints for produced summaries
            EMBEDDING_ENCODING = os.environ.get("EMBEDDING_ENCODING", "cl100k_base")
            try:
                EMBEDDING_MAX_INPUT_TOKENS = int(
                    os.environ.get("EMBEDDING_MAX_INPUT_TOKENS", "8000"),
                )
            except Exception:
                EMBEDDING_MAX_INPUT_TOKENS = 8000

            # Chunk overlap ratio for context preservation
            CHUNK_OVERLAP_RATIO = 0.2

            # Maximum number of retry attempts
            MAX_RETRY_ATTEMPTS = 3

            def _safe_map(name: str, items: list[dict], fn):
                """Run unify.map with fallback to sequential when the pool errors."""
                try:
                    return unify.map(fn, items, name=name) if items else []
                except Exception as e:
                    return [fn(**it) for it in items]

            def _semantic_chunk_text_by_tokens(
                text: str,
                max_tokens_for_text: int,
            ) -> List[str]:
                """
                Split *text* into token-aware chunks with sentence boundaries where possible.
                Each chunk will be **<= max_tokens_for_text**.
                """
                if is_within_token_limit_bytes(
                    text,
                    max_tokens_for_text,
                ):
                    return [text]

                # Split by sentences first (preserve semantic units)
                import re

                sentences = re.split(r"(?<=[.!?])\s+", text)

                chunks: List[str] = []
                current_chunk: List[str] = []
                current_tokens = 0
                overlap_buffer: List[str] = []
                overlap_tokens = int(max_tokens_for_text * CHUNK_OVERLAP_RATIO)

                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    sent = sentence.strip()
                    sent_tokens = count_tokens_per_utf_byte(sent)

                    # If single sentence exceeds limit, split by clauses
                    if sent_tokens > max_tokens_for_text:
                        # Split by common clause markers
                        clauses = re.split(r"(?<=[,;:])\s+", sent)
                        for clause in clauses:
                            if not clause.strip():
                                continue
                            clause_tokens = count_tokens_per_utf_byte(clause)
                            if current_tokens + clause_tokens > max_tokens_for_text:
                                if current_chunk:
                                    chunks.append(" ".join(current_chunk).strip())
                                    overlap_text = " ".join(current_chunk).strip()
                                    overlap_trimmed = clip_text_to_token_limit_bytes(
                                        overlap_text,
                                        overlap_tokens,
                                    )
                                    overlap_buffer = (
                                        [overlap_trimmed] if overlap_trimmed else []
                                    )
                                    current_chunk = overlap_buffer + [clause]
                                    current_tokens = count_tokens_per_utf_byte(
                                        " ".join(current_chunk),
                                    )
                                else:
                                    # Force add even if too long (rare)
                                    chunks.append(
                                        clip_text_to_token_limit_bytes(
                                            clause,
                                            max_tokens_for_text,
                                        ),
                                    )
                            else:
                                current_chunk.append(clause)
                                current_tokens += clause_tokens
                    elif current_tokens + sent_tokens > max_tokens_for_text:
                        # Complete current chunk
                        if current_chunk:
                            chunks.append(" ".join(current_chunk).strip())
                            overlap_text = " ".join(current_chunk).strip()
                            overlap_trimmed = clip_text_to_token_limit_bytes(
                                overlap_text,
                                overlap_tokens,
                            )
                            overlap_buffer = (
                                [overlap_trimmed] if overlap_trimmed else []
                            )
                        current_chunk = overlap_buffer + [sent]
                        current_tokens = count_tokens_per_utf_byte(
                            " ".join(current_chunk),
                        )
                    else:
                        current_chunk.append(sent)
                        current_tokens += sent_tokens

                # Add final chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk).strip())

                return chunks

            def _chunk_text_if_needed(
                text: str,
                prompt_builder,
                context_info: dict = None,
                max_tokens: int = INITIAL_MAX_TOKENS,
                attempt: int = 0,
            ):
                """Split text into **token-aware** chunks if needed, with progressive token limit reduction."""

                # Build the prompt first to account for its tokens
                if context_info and "chunk_number" in context_info:
                    prompt = prompt_builder(
                        context_info["chunk_number"],
                        context_info["total_chunks"],
                    )
                else:
                    prompt = prompt_builder()

                prompt_tokens = conservative_token_estimate(
                    prompt,
                    SUMMARISER_MODEL_ENCODING,
                )
                available_tokens_for_text = max(max_tokens - prompt_tokens, 256)

                if not has_meaningful_text(text):
                    return ""  # nothing to summarise sensibly

                if (
                    is_within_token_limit_bytes(
                        text,
                        available_tokens_for_text,
                    )
                    and text.strip()
                ):  # Text fits in single call
                    return generate_summary_with_compression(
                        client,
                        prompt,
                        text,
                        embedding_encoding=EMBEDDING_ENCODING,
                        max_embedding_tokens=EMBEDDING_MAX_INPUT_TOKENS,
                    )

                # Text needs chunking (token-aware)
                chunks = _semantic_chunk_text_by_tokens(text, available_tokens_for_text)

                if len(chunks) == 1:
                    # Single chunk after semantic splitting
                    return generate_summary_with_compression(
                        client,
                        prompt,
                        chunks[0],
                        embedding_encoding=EMBEDDING_ENCODING,
                        max_embedding_tokens=EMBEDDING_MAX_INPUT_TOKENS,
                    )

                # Summarize chunks in parallel

                # Prepare chunk data for parallel processing
                chunk_data = []
                for i, chunk in enumerate(chunks):
                    if not has_meaningful_text(chunk):
                        continue
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
                    # Account for prompt tokens for this chunk-specific prompt
                    chunk_prompt_tokens = conservative_token_estimate(
                        chunk_prompt,
                        SUMMARISER_MODEL_ENCODING,
                    )
                    available_for_chunk = max(max_tokens - chunk_prompt_tokens, 256)
                    chunk_text = data["chunk"]
                    if not is_within_token_limit_bytes(chunk_text, available_for_chunk):
                        chunk_text = clip_text_to_token_limit_bytes(
                            chunk_text,
                            available_for_chunk,
                        )
                    return generate_summary_with_compression(
                        client,
                        chunk_prompt,
                        chunk_text,
                        embedding_encoding=EMBEDDING_ENCODING,
                        max_embedding_tokens=EMBEDDING_MAX_INPUT_TOKENS,
                    )

                # Run chunk summaries in parallel with fallback
                chunk_summaries = _safe_map(
                    name=f"Chunk Summaries ({len(chunks)} chunks)",
                    items=[
                        {
                            "chunk": it["chunk"],
                            "chunk_num": it["chunk_num"],
                            "total_chunks": it["total_chunks"],
                        }
                        for it in chunk_data
                    ],
                    fn=_summarize_chunk,
                )

                # Combine chunk summaries
                if len(chunk_summaries) > 1:
                    combined = "\n\n".join(chunk_summaries)
                    # Account for prompt tokens before deciding next action
                    prompt_for_next = prompt_builder()
                    prompt_tokens_next = conservative_token_estimate(
                        prompt_for_next,
                        SUMMARISER_MODEL_ENCODING,
                    )
                    available_next = max(max_tokens - prompt_tokens_next, 256)
                    # Check if combined summaries need further summarization
                    if count_tokens_per_utf_byte(combined) > available_next:
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
                            clipped_combined = clip_text_to_token_limit_bytes(
                                combined,
                                available_next,
                            )
                            return generate_summary_with_compression(
                                client,
                                prompt_for_next,
                                clipped_combined,
                                embedding_encoding=EMBEDDING_ENCODING,
                                max_embedding_tokens=EMBEDDING_MAX_INPUT_TOKENS,
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
                    if para.summary is None and has_meaningful_text(para.text or ""):
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

            # Define the paragraph summary runner
            def _summarize_paragraph(**para_data):
                """Summarize a single paragraph."""
                # Account for prompt tokens before passing to chunk handler
                para_prompt = build_paragraph_summary_prompt()
                para_prompt_tokens = conservative_token_estimate(
                    para_prompt,
                    SUMMARISER_MODEL_ENCODING,
                )
                # Reuse chunk handler which recalculates; we pre-trim to be safe
                text = para_data["text"]
                available = max(INITIAL_MAX_TOKENS - para_prompt_tokens, 256)
                if not is_within_token_limit_bytes(text, available):
                    text = clip_text_to_token_limit_bytes(text, available)
                return _chunk_text_if_needed(
                    text,
                    build_paragraph_summary_prompt,
                )

            # Run all paragraph summaries in parallel
            para_summaries: list[str] = []
            if paragraphs_to_process:
                para_summaries = _safe_map(
                    name="Paragraph Summaries",
                    items=paragraphs_to_process,
                    fn=_summarize_paragraph,
                )

                # Update the document with the summaries
                for (section_idx, para_idx), summary in zip(
                    paragraph_refs,
                    para_summaries,
                ):
                    document.sections[section_idx].paragraphs[para_idx].summary = (
                        summary or ""
                    )

            # LEVEL 2: Parallel Section summaries

            # Collect all sections that need summarization
            sections_to_process = []
            section_refs = []

            for section_idx, section in enumerate(document.sections):
                if (
                    section.summary is None
                    and section.paragraphs
                    and any(
                        has_meaningful_text(p.summary or p.text or "")
                        for p in section.paragraphs
                    )
                ):  # Combine paragraph summaries
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

            # Define the section summary runner
            def _summarize_section(**section_data):
                """Summarize a single section."""
                # Account for prompt tokens before passing to chunk handler
                sec_prompt = build_section_summary_prompt()
                sec_prompt_tokens = conservative_token_estimate(
                    sec_prompt,
                    SUMMARISER_MODEL_ENCODING,
                )
                available = max(INITIAL_MAX_TOKENS - sec_prompt_tokens, 256)
                combined = section_data["combined_summaries"]
                if not is_within_token_limit_bytes(combined, available):
                    combined = clip_text_to_token_limit_bytes(combined, available)
                return _chunk_text_if_needed(
                    combined,
                    build_section_summary_prompt,
                )

            # Run all section summaries in parallel
            section_summaries: list[str] = []
            if sections_to_process:
                section_summaries = _safe_map(
                    name="Section Summaries",
                    items=sections_to_process,
                    fn=_summarize_section,
                )

                # Update the document with the summaries
                for section_idx, summary in zip(section_refs, section_summaries):
                    document.sections[section_idx].summary = summary or ""

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

                final_doc_summary = document.full_text[:EMBEDDING_MAX_INPUT_TOKENS]
                if section_summaries:
                    combined_sections = "\n\n".join(section_summaries)

                    # Add document metadata for context
                    doc_context = f"Document: {document.metadata.title or 'Untitled'}\n"
                    doc_context += f"Type: {document.metadata.file_type}\n"
                    doc_context += f"Total Sections: {len(document.sections)}\n\n"

                    # Include image annotations and table HTML
                    try:
                        image_annotations = []
                        for img in getattr(document.metadata, "images", []) or []:
                            try:
                                ann = (img.annotation or "").strip()
                            except Exception:
                                ann = ""
                            if ann:
                                image_annotations.append(ann)
                        images_block = (
                            "Image Annotations:\n" + "\n".join(image_annotations)
                            if image_annotations
                            else ""
                        )
                    except Exception:
                        images_block = ""

                    try:
                        table_html = []
                        for tbl in getattr(document.metadata, "tables", []) or []:
                            try:
                                html = (tbl.html or "").strip()
                            except Exception:
                                html = ""
                            if html:
                                table_html.append(html)
                        tables_block = (
                            "Tables (HTML):\n" + "\n\n".join(table_html)
                            if table_html
                            else ""
                        )
                    except Exception:
                        tables_block = ""

                    # Generate document summary from section summaries (exclude images/tables from input)
                    doc_prompt = build_document_summary_prompt()

                    # Ensure the input fits within summariser budget (prompt + text)
                    prompt_tokens = conservative_token_estimate(
                        doc_prompt,
                        SUMMARISER_MODEL_ENCODING,
                    )
                    budget = max(INITIAL_MAX_TOKENS - prompt_tokens, 256)
                    summary_source = "\n\n".join(
                        [part for part in [doc_context, combined_sections] if part],
                    )
                    clipped_input = clip_text_to_token_limit_bytes(
                        summary_source,
                        budget,
                    )
                    generated_summary = generate_summary_with_compression(
                        client,
                        doc_prompt,
                        clipped_input,
                        embedding_encoding=EMBEDDING_ENCODING,
                        max_embedding_tokens=EMBEDDING_MAX_INPUT_TOKENS,
                        post_generation_ctx="\n\n".join(
                            [p for p in [images_block, tables_block] if p],
                        )
                        or None,
                    )
                    final_doc_summary = generated_summary

                document.summary = final_doc_summary

            print("Summary generation completed")

        except Exception as e:
            # Fallback to basic summaries
            print("Summary generation failed, falling back to basic summaries")
            self._generate_basic_summaries(document)

    def _generate_basic_summaries(self, document: Document) -> None:
        """
        Generate **token-aware** fallback summaries when LLM summarisation is unavailable.

        Goals
        -----
        • Produce meaningful summaries for paragraphs → sections → document.
        • Ensure all outputs are embeddable by the downstream embedding model.
        • Avoid hard-coded magic numbers; use env-tunable limits and token utilities.

        Env Vars (tunable)
        -------------------
        EMBEDDING_ENCODING
            tiktoken encoding used by the embedding model (default: "cl100k_base").
            Examples: "cl100k_base" for text-embedding-3-* models.
        EMBEDDING_MAX_INPUT_TOKENS
            Maximum input tokens the embedding model should receive for **one** text.
            Default is 8000 (keeps a small safety margin under ~8.1k).

        Behaviour
        ---------
        1) Paragraph summaries: whitespace-normalised copies of the paragraph text.
        2) Section summaries: bullet list of paragraph summaries, clipped *per section*
           to a fair share of the overall embedding budget.
        3) Document summary: join of section summaries, clipped to the full budget.
        4) Defensive last-resort: ensure `document.summary` is always a non-empty string.
        """
        import os

        try:
            # Token limits for the embedding model (env-tunable; sensible defaults)
            embed_encoding = os.environ.get("EMBEDDING_ENCODING", "cl100k_base")
            # Leave a safety margin so headers/joiners never push us over model limits
            max_doc_tokens = int(os.environ.get("EMBEDDING_MAX_INPUT_TOKENS", "8000"))

            sections = document.sections or []

            # ────────────────────────────────────────────────────────────────────
            # 1) Paragraph-level summaries (no hard truncation here)
            #    Keep summaries compact & normalised; token clipping happens at higher levels
            # ────────────────────────────────────────────────────────────────────
            for section in sections:
                for para in section.paragraphs or []:
                    if para.summary is not None:
                        continue
                    raw = (para.text or "").strip()
                    if not has_meaningful_text(raw):
                        continue
                    # Whitespace-normalised text keeps length predictable for later clipping
                    para.summary = " ".join(raw.split())

            # ────────────────────────────────────────────────────────────────────
            # 2) Section-level summaries
            #    Aggregate bullet points from paragraph summaries and clip to *per-section* budget
            # ────────────────────────────────────────────────────────────────────
            sec_count = max(1, len(sections))
            # Fair share of the total embedding budget per section, with a small minimum
            per_section_cap = max(256, max_doc_tokens // sec_count)

            for section in sections:
                if section.summary is None:
                    # Build from paragraph summaries (or paragraph text when summary missing)
                    bullets: list[str] = []
                    for p in section.paragraphs or []:
                        bit = (p.summary or p.text or "").strip()
                        if has_meaningful_text(bit):
                            bullets.append(f"• {bit}")

                    # If nothing to summarise, try to at least retain a title
                    if not bullets and has_meaningful_text(section.title):
                        bullets.append(section.title.strip())

                    if bullets:
                        sec_text = "\n".join(bullets)
                        if has_meaningful_text(section.title):
                            # Prepend title; downstream clipping will enforce the cap
                            sec_text = f"{section.title.strip()}\n{sec_text}"

                        # Clip to per-section share so each section remains embeddable
                        if not is_within_token_limit_bytes(sec_text, per_section_cap):
                            sec_text = clip_text_to_token_limit_bytes(
                                sec_text,
                                per_section_cap,
                            )
                        section.summary = sec_text
                else:
                    # If section already has a summary, still enforce the per-section cap
                    if not is_within_token_limit_bytes(
                        section.summary,
                        per_section_cap,
                    ):
                        section.summary = clip_text_to_token_limit_bytes(
                            section.summary,
                            per_section_cap,
                        )

            # ────────────────────────────────────────────────────────────────────
            # 3) Document-level summary
            #    Concatenate section summaries and clip to the full budget
            # ────────────────────────────────────────────────────────────────────
            if document.summary is None:
                doc_parts: list[str] = []
                for s in sections:
                    text = (s.summary or "").strip()
                    if has_meaningful_text(text):
                        doc_parts.append(text)

                combined = "\n\n".join(doc_parts)
                if not has_meaningful_text(combined):
                    # Fall back to full_text if sections carry no usable content
                    combined = (document.full_text or "").strip()

                if has_meaningful_text(combined):
                    if not is_within_token_limit_bytes(combined, max_doc_tokens):
                        combined = clip_text_to_token_limit_bytes(
                            combined,
                            max_doc_tokens,
                        )
                    document.summary = combined

            # ────────────────────────────────────────────────────────────────────
            # 4) Final guard – never return an empty/None document summary
            # ────────────────────────────────────────────────────────────────────
            if not has_meaningful_text(document.summary):
                raw = (document.full_text or "").strip()
                if has_meaningful_text(raw):
                    try:
                        document.summary = (
                            raw
                            if is_within_token_limit_bytes(raw, max_doc_tokens)
                            else clip_text_to_token_limit_bytes(
                                raw,
                                max_doc_tokens,
                            )
                        )
                    except Exception:
                        # As a last resort, a small slice to guarantee non-empty output
                        document.summary = raw[:500].rstrip() + (
                            "…" if len(raw) > 500 else ""
                        )
                else:
                    document.summary = "Document parsing completed."

        except Exception:
            # Defensive last resort – do not raise from fallback path
            raw = (getattr(document, "full_text", None) or "").strip()
            try:
                embed_encoding = os.environ.get("EMBEDDING_ENCODING", "cl100k_base")
                max_doc_tokens = int(
                    os.environ.get("EMBEDDING_MAX_INPUT_TOKENS", "8000"),
                )
                document.summary = (
                    clip_text_to_token_limit_bytes(raw, max_doc_tokens)
                    if raw
                    else "Document parsing completed."
                )
            except Exception:
                document.summary = (
                    raw[:500].rstrip() + ("…" if len(raw) > 500 else "")
                    if raw
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
