"""
Shared fixtures for FileManager tests and parser tests.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from typing import Any, Dict, List
from unity.file_manager.parser import DoclingParser


@pytest.fixture()
def parser():
    """Create a parser instance with LLM enrichment disabled for testing."""
    return DoclingParser(use_llm_enrichment=False)


@pytest.fixture()
def parser_with_llm():
    """Create a parser instance with LLM enrichment enabled."""
    return DoclingParser(use_llm_enrichment=True)


@pytest.fixture()
def parser_with_options():
    """Create a parser instance with custom options for advanced testing."""
    return DoclingParser(
        use_llm_enrichment=False,
        max_chunk_size=256,
        chunk_overlap=50,
        use_hybrid_chunking=True,
        extract_images=True,
        extract_tables=True,
    )


def _get_format_content_generators() -> Dict[str, Dict[str, Any]]:
    """
    Get content generators for each supported format.

    Returns:
        Dict mapping file extensions to their content specifications.
    """
    return {
        ".txt": {
            "mime_type": "text/plain",
            "generator": _create_text_content,
            "validation_patterns": ["simple text", "text file"],
            "structure_expectations": {
                "min_sections": 1,
                "min_paragraphs": 1,
                "min_sentences": 1,
            },
        },
        ".md": {
            "mime_type": "text/markdown",
            "generator": _create_markdown_content,
            "validation_patterns": [
                "# Main Title",
                "## Section",
                "**bold**",
                "*italic*",
            ],
            "structure_expectations": {
                "min_sections": 2,  # Should parse headers as sections
                "min_paragraphs": 3,
                "min_sentences": 5,
            },
        },
        ".html": {
            "mime_type": "text/html",
            "generator": _create_html_content,
            "validation_patterns": ["Main Heading", "First paragraph", "emphasis"],
            "structure_expectations": {
                "min_sections": 1,
                "min_paragraphs": 2,
                "min_sentences": 3,
            },
        },
        ".csv": {
            "mime_type": "text/csv",
            "generator": _create_csv_content,
            "validation_patterns": ["Name,Age,City", "John Doe", "Jane Smith"],
            "structure_expectations": {
                "min_sections": 1,
                "min_paragraphs": 1,
                "min_sentences": 1,
            },
        },
        ".json": {
            "mime_type": "application/json",
            "generator": _create_json_content,
            "validation_patterns": ["title", "content", "metadata"],
            "structure_expectations": {
                "min_sections": 1,
                "min_paragraphs": 1,
                "min_sentences": 1,
            },
        },
        ".pdf": {
            "mime_type": "application/pdf",
            "generator": None,  # Use sample files
            "validation_patterns": [],  # Determined by actual content
            "structure_expectations": {
                "min_sections": 1,
                "min_paragraphs": 5,
                "min_sentences": 10,
            },
        },
        ".docx": {
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "generator": None,  # Use sample files
            "validation_patterns": [],  # Determined by actual content
            "structure_expectations": {
                "min_sections": 1,
                "min_paragraphs": 10,
                "min_sentences": 20,
            },
        },
    }


def _create_text_content(variant: str = "simple") -> str:
    """Generate text content for different test scenarios."""
    if variant == "simple":
        return "This is a simple text file."
    elif variant == "multi_paragraph":
        return (
            "First paragraph with important content.\n\n"
            "Second paragraph with more details and information.\n\n"
            "Third paragraph concludes the document with final thoughts."
        )
    elif variant == "special_chars":
        return "Text with special chars: café, naïve, €100, 你好世界, математика"
    elif variant == "empty":
        return ""
    elif variant == "large":
        return "\n\n".join(
            [
                f"Paragraph {i}: Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
                f"Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
                for i in range(1, 51)
            ],
        )
    elif variant == "long_lines":
        return " ".join(["word"] * 1000) + "\n\nSecond paragraph."
    elif variant == "mixed_encoding":
        return (
            "UTF-8: 你好世界\nLatin: café naïve\nCyrillic: математика\nGreek: φιλοσοφία"
        )
    else:
        return f"Test content for {variant} variant."


def _create_markdown_content(variant: str = "simple") -> str:
    """Generate markdown content for testing."""
    if variant == "simple":
        return (
            "# Main Title\n\n"
            "This is an introduction paragraph with some content.\n\n"
            "## Section 1\n\n"
            "Content for section 1 with **bold** and *italic* text.\n\n"
            "### Subsection 1.1\n\n"
            "Detailed content here with more information.\n\n"
            "## Section 2\n\n"
            "More content with a [link](https://example.com) and `code`.\n\n"
            "- List item 1\n"
            "- List item 2\n"
            "- List item 3\n"
        )
    elif variant == "complex":
        return (
            "# Document Title\n\n"
            "Introduction paragraph.\n\n"
            "## First Section\n\n"
            "Some content with **emphasis** and *italics*.\n\n"
            "```python\n"
            "def hello():\n"
            "    return 'world'\n"
            "```\n\n"
            "### Subsection A\n\n"
            "More detailed content here.\n\n"
            "#### Deep Subsection\n\n"
            "Even more nested content.\n\n"
            "## Second Section\n\n"
            "Final content with [external link](https://example.com).\n"
        )
    else:
        return f"# {variant.title()}\n\nContent for {variant} markdown test."


def _create_html_content(variant: str = "simple") -> str:
    """Generate HTML content for testing."""
    if variant == "simple":
        return (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head><title>Test Page</title></head>\n"
            "<body>\n"
            "<h1>Main Heading</h1>\n"
            "<p>First paragraph with <strong>emphasis</strong> and <em>italic</em> text.</p>\n"
            "<h2>Subheading</h2>\n"
            "<p>Second paragraph with a <a href='#'>link</a> and more content.</p>\n"
            "<ul>\n"
            "<li>Item 1</li>\n"
            "<li>Item 2</li>\n"
            "<li>Item 3</li>\n"
            "</ul>\n"
            "<p>Final paragraph with conclusion.</p>\n"
            "</body>\n"
            "</html>"
        )
    elif variant == "complex":
        return (
            "<!DOCTYPE html>\n"
            "<html>\n"
            "<head><title>Complex Test Page</title></head>\n"
            "<body>\n"
            "<h1>Document Title</h1>\n"
            "<p>Introduction with <strong>important</strong> information.</p>\n"
            "<h2>First Section</h2>\n"
            "<p>Content with <em>emphasis</em> and <code>inline code</code>.</p>\n"
            "<h3>Subsection</h3>\n"
            "<p>More detailed content here.</p>\n"
            "<table>\n"
            "<tr><th>Name</th><th>Value</th></tr>\n"
            "<tr><td>Item 1</td><td>Value 1</td></tr>\n"
            "<tr><td>Item 2</td><td>Value 2</td></tr>\n"
            "</table>\n"
            "<h2>Second Section</h2>\n"
            "<p>Final content with <a href='https://example.com'>external link</a>.</p>\n"
            "</body>\n"
            "</html>"
        )
    else:
        return f"<html><head><title>{variant}</title></head><body><h1>{variant}</h1><p>Test content</p></body></html>"


def _create_csv_content(variant: str = "simple") -> str:
    """Generate CSV content for testing."""
    if variant == "simple":
        return (
            "Name,Age,City,Country\n"
            "John Doe,30,New York,USA\n"
            "Jane Smith,25,London,UK\n"
            "Bob Johnson,35,Sydney,Australia\n"
            "Alice Brown,28,Toronto,Canada\n"
        )
    elif variant == "complex":
        return (
            "ID,Name,Email,Department,Salary,Start Date\n"
            '1,"John Doe",john.doe@company.com,Engineering,75000,2020-01-15\n'
            '2,"Jane Smith",jane.smith@company.com,Marketing,65000,2019-03-22\n'
            '3,"Bob Johnson",bob.johnson@company.com,Sales,70000,2021-07-10\n'
            '4,"Alice Brown",alice.brown@company.com,HR,60000,2020-11-05\n'
        )
    else:
        return f"Header1,Header2\nValue1,Value2\n{variant},test"


def _create_json_content(variant: str = "simple") -> str:
    """Generate JSON content for testing."""
    if variant == "simple":
        data = {
            "title": "Test Document",
            "content": "This is test JSON content for parsing.",
            "metadata": {
                "author": "Test Author",
                "created": "2024-01-01",
                "version": "1.0",
            },
        }
    elif variant == "nested":
        data = {
            "document": {
                "title": "Complex Document",
                "sections": [
                    {
                        "heading": "Introduction",
                        "content": "This is the introduction section with important information.",
                        "subsections": [
                            {"title": "Overview", "text": "Overview content"},
                            {"title": "Purpose", "text": "Purpose content"},
                        ],
                    },
                    {
                        "heading": "Main Content",
                        "content": "This is the main content section.",
                        "data": ["item1", "item2", "item3"],
                    },
                ],
                "metadata": {
                    "author": "Complex Author",
                    "tags": ["test", "parsing", "json"],
                    "word_count": 150,
                },
            },
        }
    elif variant == "array":
        data = [
            {"id": 1, "name": "First Item", "description": "Description of first item"},
            {
                "id": 2,
                "name": "Second Item",
                "description": "Description of second item",
            },
            {"id": 3, "name": "Third Item", "description": "Description of third item"},
        ]
    else:
        data = {"variant": variant, "content": f"Test content for {variant}"}

    return json.dumps(data, indent=2)


def _create_sample_file(file_path: Path, file_type: str) -> None:
    """Create a sample file of the specified type with appropriate content."""
    if file_type == ".txt":
        file_path.write_text(
            "Sample Document\n\n"
            "This is a test document with multiple paragraphs.\n\n"
            "Each paragraph contains important information that should be parsed correctly.",
            encoding="utf-8",
        )
    elif file_type in [".pdf", ".docx"]:
        # For binary formats like PDF and DOCX, we cannot create valid files with text content
        # These would need actual binary content or libraries to create them
        # For now, skip creating these files in basic testing scenarios
        # They can be tested separately with actual binary files
        return
    else:
        # Fallback for any other supported format
        file_path.write_text(
            f"Sample content for {file_type} file format.",
            encoding="utf-8",
        )


@pytest.fixture()
def sample_files(tmp_path: Path) -> Path:
    """Create sample files for all supported formats."""
    from unity.file_manager.file_manager import FileManager

    d = tmp_path / "samples"
    d.mkdir(parents=True, exist_ok=True)

    # Get supported formats from the current FileManager/parser
    fm = FileManager()
    supported_formats = fm.supported_formats

    # Create sample files for each supported format
    for i, fmt in enumerate(supported_formats):
        if fmt in [".pdf", ".docx"]:
            # Skip binary formats that need special handling
            continue
        filename = f"sample_{i}{fmt}"
        file_path = d / filename
        _create_sample_file(file_path, fmt)

    # Always create an empty file for empty file testing
    (d / "empty.txt").touch()

    # Create a second file for multi-file testing
    if ".txt" in supported_formats:
        (d / "notes.txt").write_text(
            "Important Notes\n\n"
            "Remember to test all functionality.\n\n"
            "This file has different content for testing purposes.",
            encoding="utf-8",
        )

    return d


@pytest.fixture()
def supported_file_examples(tmp_path: Path) -> dict:
    """Create examples of files in each supported format with expected content."""
    from unity.file_manager.file_manager import FileManager

    fm = FileManager()
    supported_formats = fm.supported_formats

    examples = {}

    # Get the path to the actual sample files directory
    current_dir = Path(__file__).parent
    sample_dir = current_dir / "sample"

    # For each supported format, find corresponding sample files or create test files
    for fmt in supported_formats:
        # First try to find existing sample files for this format
        sample_files = list(sample_dir.glob(f"*{fmt}")) if sample_dir.exists() else []

        if sample_files:
            # Use the first available sample file for this format
            sample_file = sample_files[0]
            examples[sample_file.name] = {
                "path": sample_file,
                "format": fmt,
                "expected_phrases": [],  # To be determined by test execution
                "is_sample_file": True,
            }
        else:
            # Create a test file for this format if no samples exist
            d = tmp_path / "examples"
            d.mkdir(parents=True, exist_ok=True)

            filename = f"test_file{fmt}"
            file_path = d / filename

            # Create appropriate content based on format
            if fmt in [".txt", ".md", ".log"]:
                content = f"Sample {fmt.upper()} Document\n\nThis contains test content for {fmt} format."
                file_path.write_text(content, encoding="utf-8")
                expected_phrases = [f"Sample {fmt.upper()} Document", "test content"]
            else:
                # For other text-based formats or fallback
                content = f"Test content for {fmt} format."
                file_path.write_text(content, encoding="utf-8")
                expected_phrases = ["Test content", fmt]

            examples[filename] = {
                "path": file_path,
                "format": fmt,
                "expected_phrases": expected_phrases,
                "is_sample_file": False,
            }

    return examples


@pytest.fixture()
def supported_format_files(tmp_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Create comprehensive test files for all supported formats.

    Returns:
        Dict mapping format extensions to file information and validation data.
    """
    parser = DoclingParser()
    supported_formats = parser.supported_formats
    format_specs = _get_format_content_generators()

    format_files = {}

    # Get sample directory for binary files
    current_dir = Path(__file__).parent
    sample_dir = current_dir / "sample"

    for fmt in supported_formats:
        if fmt not in format_specs:
            continue

        spec = format_specs[fmt]

        if spec["generator"] is None:
            # Handle binary formats - use sample files if available
            if sample_dir.exists():
                sample_files = list(sample_dir.glob(f"*{fmt}"))
                if sample_files:
                    sample_file = sample_files[0]
                    format_files[fmt] = {
                        "files": {"sample": sample_file},
                        "mime_type": spec["mime_type"],
                        "validation_patterns": spec["validation_patterns"],
                        "structure_expectations": spec["structure_expectations"],
                        "is_binary": True,
                        "variants": ["sample"],
                    }
            continue

        # Create test files for text-based formats
        format_dir = tmp_path / f"format_{fmt[1:]}"  # Remove the dot
        format_dir.mkdir(parents=True, exist_ok=True)

        # Create multiple variants for thorough testing
        variants = (
            ["simple", "complex", "empty"]
            if fmt != ".txt"
            else [
                "simple",
                "multi_paragraph",
                "special_chars",
                "empty",
                "large",
                "long_lines",
            ]
        )

        files = {}
        for variant in variants:
            if variant == "empty":
                content = ""
            else:
                content = spec["generator"](variant)

            filename = f"test_{variant}{fmt}"
            file_path = format_dir / filename

            if content:
                file_path.write_text(content, encoding="utf-8")
            else:
                file_path.touch()  # Create empty file

            files[variant] = file_path

        format_files[fmt] = {
            "files": files,
            "mime_type": spec["mime_type"],
            "validation_patterns": spec["validation_patterns"],
            "structure_expectations": spec["structure_expectations"],
            "is_binary": False,
            "variants": variants,
        }

    return format_files


@pytest.fixture()
def edge_case_files(tmp_path: Path) -> Dict[str, Path]:
    """Create files for edge case testing."""
    edge_dir = tmp_path / "edge_cases"
    edge_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # Very large file
    large_content = "\n\n".join([f"Section {i}: " + "word " * 500 for i in range(100)])
    files["very_large"] = edge_dir / "very_large.txt"
    files["very_large"].write_text(large_content, encoding="utf-8")

    # File with only whitespace
    files["whitespace_only"] = edge_dir / "whitespace.txt"
    files["whitespace_only"].write_text("   \n\t  \n   ", encoding="utf-8")

    # File with very long lines
    files["long_lines"] = edge_dir / "long_lines.txt"
    files["long_lines"].write_text("x" * 10000 + "\n" + "y" * 10000, encoding="utf-8")

    # File with mixed encodings (UTF-8)
    files["mixed_unicode"] = edge_dir / "unicode.txt"
    files["mixed_unicode"].write_text(
        "English text\n"
        "中文内容\n"
        "العربية\n"
        "Русский текст\n"
        "Ελληνικά\n"
        "हिन्दी\n",
        encoding="utf-8",
    )

    # Malformed JSON
    files["malformed_json"] = edge_dir / "malformed.json"
    files["malformed_json"].write_text('{"incomplete": "json"', encoding="utf-8")

    # Binary content in text file
    files["binary_in_text"] = edge_dir / "binary.txt"
    files["binary_in_text"].write_bytes(b"\x00\x01\x02\xff\xfe\xfd" + b"text content")

    return files


@pytest.fixture()
def parser_validation_suite() -> Dict[str, Any]:
    """
    Provide validation functions and expectations for parser testing.

    Returns:
        Dict containing validation functions and standard expectations.
    """

    def validate_document_structure(document, expectations: Dict[str, int]):
        """Validate basic document structure meets expectations."""
        assert len(document.sections) >= expectations.get("min_sections", 0)

        total_paragraphs = sum(len(section.paragraphs) for section in document.sections)
        assert total_paragraphs >= expectations.get("min_paragraphs", 0)

        total_sentences = sum(
            len(para.sentences)
            for section in document.sections
            for para in section.paragraphs
        )
        assert total_sentences >= expectations.get("min_sentences", 0)

    def validate_metadata(document, expected_mime_type: str):
        """Validate document metadata is properly set."""
        assert document.metadata is not None
        assert document.metadata.file_type == expected_mime_type
        assert document.metadata.parser_name == "DoclingParser"
        assert document.metadata.processing_time is not None
        assert document.processing_status == "completed"

    def validate_content_preservation(document, validation_patterns: List[str]):
        """Validate that important content patterns are preserved."""
        full_text = document.to_plain_text()
        for pattern in validation_patterns:
            assert (
                pattern.lower() in full_text.lower()
            ), f"Pattern '{pattern}' not found in parsed content"

    def validate_flat_records(records: List[Dict[str, Any]]):
        """Validate flat record structure and content."""
        assert len(records) > 0, "Should produce at least one record"

        # Check for document record
        doc_records = [r for r in records if r.get("content_type") == "document"]
        assert len(doc_records) == 1, "Should have exactly one document record"

        # Validate record structure
        required_fields = [
            "content_id",
            "content_type",
            "title",
            "summary",
            "content_text",
            "document_id",
            "level",
            "confidence_score",
            "schema_id",
        ]
        for record in records:
            for field in required_fields:
                assert field in record, f"Required field '{field}' missing from record"

            assert record["content_type"] in [
                "document",
                "section",
                "paragraph",
                "sentence",
            ]
            assert isinstance(record["confidence_score"], (int, float))
            assert 0.0 <= record["confidence_score"] <= 1.0

    return {
        "validate_structure": validate_document_structure,
        "validate_metadata": validate_metadata,
        "validate_content": validate_content_preservation,
        "validate_records": validate_flat_records,
        "standard_expectations": {
            "min_processing_time": 0.0,
            "max_processing_time": 30.0,  # seconds
            "min_confidence": 0.0,
            "max_confidence": 1.0,
        },
    }


@pytest.fixture()
def performance_benchmarks() -> Dict[str, Any]:
    """Provide performance benchmarks for parser testing."""
    return {
        "max_parse_time": {
            "small_file": 5.0,  # < 1KB
            "medium_file": 15.0,  # 1KB - 100KB
            "large_file": 60.0,  # > 100KB
        },
        "memory_limits": {
            "max_memory_mb": 500,  # Maximum memory usage during parsing
        },
        "batch_limits": {
            "max_batch_time": 120.0,  # seconds for batch processing
            "min_concurrency": 2,  # minimum concurrent files
        },
    }


# Import the proper SimulatedFileManager
from unity.file_manager.simulated import SimulatedFileManager


@pytest.fixture()
def simulated_file_manager() -> SimulatedFileManager:
    """Create a simulated FileManager for unit tests."""
    return SimulatedFileManager()
