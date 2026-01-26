"""
Shared fixtures for FileManager tests and parser tests.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from typing import Any, Dict
from unity.file_manager.managers.local import LocalFileManager
from unity.file_manager.global_file_manager import GlobalFileManager
from unity.file_manager.simulated import SimulatedFileManager
from unity.common.llm_client import new_llm_client


async def llm_judge_html_equivalence(
    expected_html: str,
    parsed_html: str,
) -> tuple[bool, str]:
    """
    Use a small LLM to judge if two HTML tables are semantically equivalent.

    This allows for structural differences (e.g., header rows wrapped in <tbody> vs not)
    as long as the content, information, and overall table structure are identical.

    Args:
        expected_html: The expected HTML table (from pandas)
        parsed_html: The parsed HTML table (from Docling)

    Returns:
        Tuple of (is_equivalent: bool, explanation: str)
    """
    try:
        pass

        system_prompt = """You are an HTML table comparison expert. Your task is to determine if two HTML tables are semantically equivalent.

Rules for comparison:
1. The actual TABLE DATA (cell contents, values, text) MUST be identical (with exceptions below)
2. The table structure (rows, columns, headers) MUST be the same
3. Minor structural differences are acceptable if they don't change the meaning:
   - Header rows inside or outside <tbody> tags
   - Presence or absence of <thead> wrappers
   - Different tag nesting that preserves the same logical structure
4. The number of rows and columns MUST match
5. Header cells and data cells must be in the correct positions
6. **CRITICAL: Missing/empty data representations are equivalent:**
   - Empty cells, "nan", "NaN", "None", blank strings, and missing values are ALL considered equivalent
   - A cell with "nan" is the same as an empty cell or a cell with whitespace
   - Different representations of missing data should NOT cause a mismatch
7. **CRITICAL: Numeric equivalence:**
   - Integers and floats representing the same value are equivalent: "30" == "30.0" == "30.00"
   - Trailing zeros in floats should be ignored: "1.5" == "1.50"
   - Different numeric formats representing the same value are acceptable
8. **CRITICAL: Parsed table can be MORE complete:**
   - If the parsed table has actual data where the expected table has missing/nan values, this is ACCEPTABLE
   - The parsed table having MORE information than expected (filling in nan/missing cells) is GOOD
   - Only flag as NOT_EQUIVALENT if parsed table is MISSING data that exists in expected table

Respond with ONLY one of these two formats:
- If tables are semantically equivalent: "EQUIVALENT: <brief reason>"
- If tables are NOT equivalent: "NOT_EQUIVALENT: <specific difference found>"
"""

        user_prompt = f"""Compare these two HTML tables:

Expected HTML:
{expected_html}

Parsed HTML:
{parsed_html}

Your response:"""

        client = new_llm_client()
        client.set_system_message(system_prompt)
        result = await client.generate(user_prompt)

        # Handle the response - it might be a string or None
        if result is None:
            return (
                expected_html == parsed_html,
                "LLM returned None, using exact string match",
            )

        response_text = str(result).strip()

        if response_text.startswith("EQUIVALENT"):
            return True, response_text.replace("EQUIVALENT:", "").strip()
        else:
            return False, response_text.replace("NOT_EQUIVALENT:", "").strip()

    except ImportError:
        # If unify is not available, fall back to string comparison
        return (
            expected_html == parsed_html,
            "LLM judge not available, using exact string match",
        )
    except Exception as e:
        # If LLM call fails, fall back to string comparison
        return (
            expected_html == parsed_html,
            f"LLM judge failed: {str(e)}, using exact string match",
        )


@pytest.fixture(scope="session")
def fm_root(tmp_path_factory) -> str:
    """Session-scoped root directory for the singleton LocalFileManager.

    All tests must instantiate the manager with this same root to respect
    singleton semantics across the suite.
    """
    return tmp_path_factory.mktemp("fm_root").as_posix()


@pytest.fixture()
def file_manager(fm_root: str) -> LocalFileManager:
    """Return the singleton LocalFileManager bound to the session's fm_root."""
    return LocalFileManager(fm_root)


@pytest.fixture()
def rootless_file_manager():
    """Provide a non-singleton FileManager with a rootless Local adapter.

    This allows tests to exercise absolute-path behavior without conflicting
    with the session-scoped LocalFileManager singleton.
    """
    from unity.file_manager.managers.file_manager import FileManager
    from unity.file_manager.filesystem_adapters.local_adapter import (
        LocalFileSystemAdapter,
    )

    return FileManager(adapter=LocalFileSystemAdapter(None))


@pytest.fixture()
def global_file_manager(file_manager):
    """Return the singleton GlobalFileManager configured with the local manager."""
    local = file_manager
    gfm = GlobalFileManager([local])
    return gfm


@pytest.fixture(scope="module")
def simulated_fm() -> SimulatedFileManager:
    """Provide a singleton-ish simulated file manager for this module."""
    return SimulatedFileManager("Demo file storage for unit-tests.")


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
                "min_paragraphs": 0,  # CSV files don't produce paragraphs, they produce tables
                "min_sentences": 0,  # CSV files don't have sentences in traditional sense
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
def sample_files(tmp_path: Path, fm_root: str) -> Path:
    """Create sample files for all supported formats."""
    d = tmp_path / "samples"
    d.mkdir(parents=True, exist_ok=True)

    # Minimal set of formats we can synthesize in tests.
    # Binary formats (pdf/docx/xlsx) are exercised via real sample files under `tests/file_manager/sample/`.
    supported_formats = [".txt", ".csv"]

    # Create sample files for each supported format
    for i, fmt in enumerate(supported_formats):
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
def supported_file_examples(tmp_path: Path, fm_root: str) -> dict:
    """Create examples of files in each supported format with expected content."""
    examples = {}

    # Get the path to the actual sample files directory
    current_dir = Path(__file__).parent
    sample_dir = current_dir / "sample"

    # Prefer using real sample files when available.
    if sample_dir.exists():
        for p in sorted(sample_dir.iterdir()):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext not in {".pdf", ".docx", ".xlsx", ".csv"}:
                continue
            examples[p.name] = {
                "path": p,
                "format": ext,
                "expected_phrases": [],
                "is_sample_file": True,
            }

    # Always add a small synthetic text file (absolute path).
    d = tmp_path / "examples"
    d.mkdir(parents=True, exist_ok=True)
    txt = d / "example.txt"
    txt.write_text(
        "This contains test content for txt format.",
        encoding="utf-8",
    )
    examples[txt.name] = {
        "path": txt,
        "format": ".txt",
        # Only check for "test content" - the exact format may vary due to LLM cache
        "expected_phrases": ["test content"],
        "is_sample_file": False,
    }

    return examples


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
