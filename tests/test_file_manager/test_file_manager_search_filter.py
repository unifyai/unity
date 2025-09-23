"""
Test FileManager search, filter, and file creation functionality.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from unity.file_manager.file_manager import FileManager
from unity.file_manager.types.file import File
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_create_file_basic():
    """Test basic _create_file functionality."""
    file_manager = FileManager()
    result = file_manager._create_file(
        filename="travel_notes.pdf",
        status="success",
        full_text="This document contains notes about hiking trails in the Alps.",
        metadata={"file_size": 2048, "file_type": "pdf", "topic": "travel"},
    )

    assert result["outcome"] == "file created successfully"
    assert "file_id" in result["details"]
    assert result["details"]["filename"] == "travel_notes.pdf"


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_create_file_with_error():
    """Test _create_file with error status."""
    file_manager = FileManager()
    result = file_manager._create_file(
        filename="corrupt_music_score.pdf",
        status="error",
        error="Failed to parse sheet music",
        full_text="",
        metadata={"topic": "music"},
    )

    assert result["outcome"] == "file created successfully"
    assert result["details"]["filename"] == "corrupt_music_score.pdf"


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_files_single_reference_basic():
    """Test basic semantic search over file contents."""
    # Create test files with different content
    file_manager = FileManager()
    files_data = [
        (
            "history_essay.pdf",
            "This document explores the causes of the French Revolution.",
        ),
        (
            "garden_guide.txt",
            "A step by step guide to planting roses and maintaining gardens.",
        ),
        ("sports_report.docx", "Summary of the Olympic Games and medal counts."),
        (
            "astronomy_notes.pdf",
            "Observations of the Orion constellation and nearby nebulae.",
        ),
    ]

    for filename, content in files_data:
        file_manager._create_file(
            filename=filename,
            status="success",
            full_text=content,
            metadata={"file_size": len(content), "file_type": filename.split(".")[-1]},
        )

    # Search for chocolate cookies related content
    query = "olympic medals"
    results = file_manager._search_files(references={"full_text": query}, k=3)

    assert len(results) >= 1
    assert isinstance(results[0], File)
    # Should find AI research doc first
    assert results[0].filename == "sports_report.docx"

    # Verify columns were created
    cols = file_manager._list_columns()
    assert "_full_text_emb" in cols


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_files_multi_columns():
    """Test semantic search across multiple columns."""
    file_manager = FileManager()
    file_manager._create_file(
        filename="wildlife_guide.pdf",
        status="success",
        full_text="Information about African elephants and their migratory patterns.",
        metadata={"file_type": "pdf", "category": "nature reference"},
        description="Information about African elephants and their migratory patterns.",
    )

    file_manager._create_file(
        filename="theatre_review.docx",
        status="success",
        full_text="Critical analysis of Shakespeare's Hamlet performances.",
        metadata={"file_type": "docx", "category": "arts review"},
        description="Critical analysis of Shakespeare's Hamlet performances.",
    )

    # Search using both content and metadata
    refs = {
        "full_text": "Shakespeare performances",
        "description": "Hamlet analysis",
    }
    results = file_manager._search_files(references=refs, k=2)

    assert len(results) >= 1
    assert results[0].filename == "theatre_review.docx"

    # Verify columns were created
    cols = file_manager._list_columns()
    assert "_full_text_emb" in cols
    assert "_description_emb" in cols


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_files_ranking_precision_k1():
    """Test that search correctly ranks documents and returns most relevant first (k=1)."""
    # Create multiple documents with different relevance levels for the same query
    file_manager = FileManager()
    test_docs = [
        (
            "ai_overview.txt",
            "Artificial intelligence is mentioned briefly in this general overview document.",
            "Artificial intelligence overview",
        ),
        (
            "ml_research.pdf",
            "Machine learning and artificial intelligence research paper discussing deep neural networks, algorithms, and AI applications in detail.",
            "Machine learning research",
        ),
        (
            "cooking_recipe.txt",
            "This chocolate chip cookie recipe has nothing to do with technology.",
            "Chocolate chip cookie recipe",
        ),
        (
            "tech_news.docx",
            "Technology news mentions AI and machine learning in passing.",
            "Technology news",
        ),
    ]
    for filename, content, description in test_docs:
        file_manager._create_file(
            filename=filename,
            status="success",
            full_text=content,
            metadata={"file_size": len(content)},
            description=description,
        )

    # Search for AI/ML content - should rank ml_research.pdf highest
    query = "artificial intelligence machine learning research algorithms"
    results = file_manager._search_files(references={"full_text": query}, k=1)

    assert len(results) == 1
    assert results[0].filename == "ml_research.pdf"

    # Verify it contains the expected content
    assert "research paper" in results[0].full_text
    assert "neural networks" in results[0].full_text


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_files_ranking_precision_k3():
    """Test search ranking with k=3 to verify correct ordering."""
    # Create documents with varying degrees of relevance
    file_manager = FileManager()
    test_docs = [
        (
            "irrelevant.txt",
            "This document is about gardening and has no technical content.",
        ),
        (
            "somewhat_relevant.docx",
            "This business document mentions AI tools in one paragraph.",
        ),
        (
            "highly_relevant.pdf",
            "Comprehensive artificial intelligence and machine learning guide with detailed algorithms, neural network architectures, and practical AI applications.",
        ),
        (
            "moderately_relevant.txt",
            "Technical article about artificial intelligence applications in industry.",
        ),
        (
            "unrelated.docx",
            "Financial report with quarterly earnings and budget forecasts.",
        ),
    ]

    for filename, content in test_docs:
        file_manager._create_file(
            filename=filename,
            status="success",
            full_text=content,
        )

    # Search for AI content
    query = "artificial intelligence machine learning algorithms and AI applications"
    results = file_manager._search_files(references={"full_text": query}, k=3)

    assert len(results) >= 3

    # First result should be the most comprehensive/relevant
    assert results[0].filename == "highly_relevant.pdf"

    # Second and third should be more relevant than irrelevant docs
    top_3_filenames = [r.filename for r in results[:3]]
    assert "irrelevant.txt" not in top_3_filenames
    assert "unrelated.docx" not in top_3_filenames


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_files_exact_match_beats_partial():
    """Test that exact keyword matches rank higher than partial matches."""
    file_manager = FileManager()
    test_docs = [
        (
            "partial_match.txt",
            "This document discusses learning and intelligent systems in general terms.",
        ),
        (
            "exact_match.pdf",
            "Machine learning artificial intelligence neural networks deep learning algorithms.",
        ),
        (
            "weak_match.docx",
            "Brief mention of smart technology and automated learning processes.",
        ),
    ]

    for filename, content in test_docs:
        file_manager._create_file(
            filename=filename,
            status="success",
            full_text=content,
            metadata={},
        )

    # Search for exact terms that appear in exact_match.pdf
    query = "machine learning artificial intelligence neural networks"
    results = file_manager._search_files(references={"full_text": query}, k=1)

    assert len(results) == 1
    assert results[0].filename == "exact_match.pdf"


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_files_multiple_reference_columns():
    """Test search across multiple reference columns with correct ranking."""
    # Create files where different signals appear in different fields
    file_manager = FileManager()
    file_manager._create_file(
        filename="signal_in_text.pdf",
        status="success",
        full_text="Deep learning neural networks for computer vision and natural language processing.",
        metadata={"category": "general", "keywords": "basic"},
        description="Deep learning neural networks for computer vision and natural language processing.",
    )

    file_manager._create_file(
        filename="signal_in_both.docx",
        status="success",
        full_text="Advanced machine learning techniques including neural networks and deep learning algorithms.",
        metadata={
            "category": "machine learning research",
            "keywords": "neural networks deep learning",
        },
        description="Advanced machine learning research including neural networks and deep learning algorithms.",
    )

    file_manager._create_file(
        filename="signal_in_metadata.txt",
        status="success",
        full_text="This is a general technology document.",
        metadata={
            "category": "deep learning neural networks",
            "keywords": "machine learning AI",
        },
        description="This is a general technology document.",
    )

    # Search with multiple reference columns - signal_in_both should rank highest
    # as it has relevant content in both full_text and metadata
    refs = {
        "full_text": "neural networks deep learning algorithms",
        "description": "machine learning research",
    }
    results = file_manager._search_files(references=refs, k=1)

    assert len(results) == 1
    assert results[0].filename == "signal_in_both.docx"

    # Verify columns were created
    cols = file_manager._list_columns()
    assert "_full_text_emb" in cols
    assert "_description_emb" in cols


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_files_domain_specific_ranking():
    """Test search ranking for domain-specific queries."""
    # Create documents in different domains
    file_manager = FileManager()
    test_docs = [
        (
            "medical_ai.pdf",
            "Artificial intelligence applications in medical diagnosis, healthcare AI systems, and clinical machine learning for patient care.",
        ),
        (
            "finance_ai.docx",
            "AI applications in financial services, algorithmic trading, and machine learning for risk assessment in banking.",
        ),
        (
            "general_ai.txt",
            "General overview of artificial intelligence and machine learning concepts.",
        ),
        (
            "automotive_ai.pdf",
            "Automotive AI systems, self-driving cars, machine learning for autonomous vehicles and transportation.",
        ),
    ]

    for filename, content in test_docs:
        file_manager._create_file(
            filename=filename,
            status="success",
            full_text=content,
            metadata={},
        )

    # Search for medical AI - should rank medical_ai.pdf first
    medical_query = "artificial intelligence medical diagnosis healthcare clinical"
    medical_results = file_manager._search_files(
        references={"full_text": medical_query},
        k=1,
    )

    assert len(medical_results) == 1
    assert medical_results[0].filename == "medical_ai.pdf"

    # Search for automotive AI - should rank automotive_ai.pdf first
    auto_query = "artificial intelligence automotive self-driving autonomous vehicles"
    auto_results = file_manager._search_files(references={"full_text": auto_query}, k=1)

    assert len(auto_results) == 1
    assert auto_results[0].filename == "automotive_ai.pdf"


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_filter_files_basic():
    """Test basic filtering of files."""
    # Create test files
    file_manager = FileManager()
    file_manager._create_file(
        filename="document.pdf",
        status="success",
        full_text="PDF content",
        metadata={"file_size": 2048},
    )

    file_manager._create_file(
        filename="spreadsheet.xlsx",
        status="error",
        error="Parse failed",
        full_text="",
        metadata={"file_size": 512},
    )

    # Filter by status
    success_files = file_manager._filter_files(filter="status == 'success'")
    assert len(success_files) >= 1
    assert all(f.status == "success" for f in success_files)
    assert isinstance(success_files[0], File)

    # Filter by filename extension
    pdf_files = file_manager._filter_files(filter="filename.endswith('.pdf')")
    assert len(pdf_files) >= 1
    assert all(f.filename.endswith(".pdf") for f in pdf_files)


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_filter_files_metadata():
    """Test filtering files by metadata fields."""
    file_manager = FileManager()
    file_manager._create_file(
        filename="large_file.pdf",
        status="success",
        full_text="Large document content",
        metadata={"file_size": 5000000, "file_type": "pdf"},
    )

    file_manager._create_file(
        filename="small_file.txt",
        status="success",
        full_text="Small text content",
        metadata={"file_size": 1000, "file_type": "txt"},
    )

    # Filter by file size
    large_files = file_manager._filter_files(filter="metadata['file_size'] > 1000000")
    assert len(large_files) >= 1
    assert large_files[0].filename == "large_file.pdf"


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_files_no_results_backfill():
    """Test that search falls back to recent files when no semantic matches."""
    file_manager = FileManager()
    file_manager._create_file(
        filename="random_doc.txt",
        status="success",
        full_text="Random content about nothing relevant",
        metadata={},
    )

    # Search for something completely unrelated
    results = file_manager._search_files(
        references={"full_text": "quantum physics molecules"},
        k=5,
    )

    # Should still return files (backfill behavior)
    assert len(results) >= 1
    assert results[0].filename == "random_doc.txt"


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_list_columns():
    """Test _list_columns functionality."""
    # Create a file to ensure table exists
    file_manager = FileManager()
    file_manager._create_file(
        filename="test.txt",
        status="success",
        full_text="test content",
        metadata={},
    )

    # Test with types
    cols_with_types = file_manager._list_columns(include_types=True)
    assert isinstance(cols_with_types, dict)
    assert "filename" in cols_with_types
    assert "status" in cols_with_types
    assert "full_text" in cols_with_types

    # Test without types
    cols_list = file_manager._list_columns(include_types=False)
    assert isinstance(cols_list, list)
    assert "filename" in cols_list


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_integration_parse_and_search(supported_file_examples: dict):
    """Test integration between parsing and search functionality."""
    # Import and parse the actual sample documents (PDF/DOCX)
    sample_paths = [
        info["path"]
        for info in supported_file_examples.values()
        if info.get("is_sample_file", False)
    ]
    assert len(sample_paths) >= 2, "Expected IT policy PDF and SmartHome DOCX files"
    file_manager = FileManager()
    added_files = []
    for sample_path in sample_paths:
        display_name = file_manager.import_file(sample_path)
        added_files.append(display_name)

    # Parse the files (this should create file records automatically)
    parse_results = file_manager.parse(added_files)

    # Verify files were logged to the table
    all_files = file_manager._filter_files()
    assert len(all_files) >= len(added_files)

    # Search for specific content from the sample files
    # Based on the provided sample content, search for IT-related terms
    it_results = file_manager._search_files(
        references={"full_text": "IT department policy security"},
        k=3,
    )

    # Should find the IT policy document
    assert len(it_results) >= 1

    # Search for SmartHome content
    smarthome_results = file_manager._search_files(
        references={"full_text": "SmartHome Hub IoT devices"},
        k=3,
    )

    # Should find the SmartHome documentation
    assert len(smarthome_results) >= 1


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_sample_files_search_ranking_k1(supported_file_examples: dict):
    """Test search ranking with actual sample files - IT vs SmartHome content."""
    # Import and parse the actual sample documents (PDF/DOCX)
    file_manager = FileManager()
    sample_paths = [
        info["path"]
        for info in supported_file_examples.values()
        if info.get("is_sample_file", False)
    ]
    assert len(sample_paths) >= 2, "Expected IT policy PDF and SmartHome DOCX files"

    added_files = []
    for sample_path in sample_paths:
        display_name = file_manager.import_file(sample_path)
        added_files.append(display_name)

    # Parse the files (this should create file records automatically)
    parse_results = file_manager.parse(added_files)

    # Verify we have both sample files
    assert len(added_files) >= 2

    # Test IT-specific search - should rank IT policy document first
    it_query = "IT governance security policy compliance GDPR"
    it_results = file_manager._search_files(
        references={"full_text": it_query},
        k=1,
    )

    assert len(it_results) == 1
    # Should get the IT policy document (contains "IT Department Policy Document")
    it_filename = it_results[0].filename.lower()
    assert "it" in it_filename or "policy" in it_filename

    # Test IoT/SmartHome specific search - should rank SmartHome doc first
    smarthome_query = "SmartHome Hub IoT devices Zigbee Z-Wave connectivity"
    smarthome_results = file_manager._search_files(
        references={"full_text": smarthome_query},
        k=1,
    )

    assert len(smarthome_results) == 1
    # Should get the SmartHome documentation
    smarthome_filename = smarthome_results[0].filename.lower()
    assert (
        "smarthome" in smarthome_filename
        or "hub" in smarthome_filename
        or "x200" in smarthome_filename
    )

    # Test technical architecture search - should rank SmartHome doc first
    # (since it has more detailed technical specs)
    tech_query = "architecture specifications processor memory connectivity API"
    tech_results = file_manager._search_files(
        references={"full_text": tech_query},
        k=1,
    )

    assert len(tech_results) == 1
    # Should prioritize the SmartHome doc which has detailed technical specs
    tech_filename = tech_results[0].filename.lower()
    assert (
        "smarthome" in tech_filename
        or "hub" in tech_filename
        or "x200" in tech_filename
    )

    # Test security-focused search - should rank IT policy first
    # (since it has comprehensive security policies)
    security_query = "security policies access control data classification encryption"
    security_results = file_manager._search_files(
        references={"full_text": security_query},
        k=1,
    )

    assert len(security_results) == 1
    # Should prioritize the IT policy doc which has detailed security policies
    security_filename = security_results[0].filename.lower()
    assert "it" in security_filename or "policy" in security_filename


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_import_file_method(sample_files: Path):
    """Test the public import_file method with basic sample files."""
    # Get a basic sample file (this test is fine with simple .txt files)
    file_manager = FileManager()
    sample_file = list(sample_files.iterdir())[0]

    # Import single file
    file_manager = FileManager()
    display_name = file_manager.import_file(sample_file)

    assert display_name is not None
    assert file_manager.exists(display_name)
    assert display_name in file_manager.list()


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_unique_filename_handling():
    """Test that unique filename logic works with file creation."""
    # Create two files with same base name (use unique name to avoid conflicts with other tests)
    file_manager = FileManager()
    file_manager._create_file(
        filename="unique_test_doc.pdf",
        status="success",
        full_text="First document",
        metadata={},
    )

    # This should work without conflict since unique names are established before parsing
    file_manager._create_file(
        filename="unique_test_doc (1).pdf",
        status="success",
        full_text="Second document",
        metadata={},
    )

    # Both files should exist
    files = file_manager._filter_files(filter="filename.startswith('unique_test_doc')")
    assert len(files) == 2

    filenames = [f.filename for f in files]
    assert "unique_test_doc.pdf" in filenames
    assert "unique_test_doc (1).pdf" in filenames
