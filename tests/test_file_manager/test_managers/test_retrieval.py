"""
Test FileManager search, filter, and file creation functionality.
"""

from __future__ import annotations


import pytest
from unity.file_manager.types.file import File
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_create_file_basic(file_manager):
    """Test basic _create_file functionality."""
    file_manager = file_manager
    result = file_manager._create_file(
        filename="travel_notes.pdf",
        status="success",
        records=[
            {
                "content_type": "document",
                "content_text": "This document contains notes about hiking trails in the Alps.",
            },
            {
                "content_type": "paragraph",
                "content_text": "Hiking in the Alps is a wonderful experience.",
            },
        ],
        full_text="This document contains notes about hiking trails in the Alps.",
        metadata={"file_size": 2048, "file_type": "pdf", "topic": "travel"},
    )

    assert result["outcome"] == "file created successfully"
    assert "file_id" in result["details"]
    assert result["details"]["filename"] == "travel_notes.pdf"


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_create_file_with_error(file_manager):
    """Test _create_file with error status."""
    file_manager = file_manager
    result = file_manager._create_file(
        filename="corrupt_music_score.pdf",
        status="error",
        error="Failed to parse sheet music",
        records=[],
        full_text="",
        metadata={"topic": "music"},
    )

    assert result["outcome"] == "file created successfully"
    assert result["details"]["filename"] == "corrupt_music_score.pdf"


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_files_single_reference_basic(file_manager):
    """Test basic semantic search over file contents."""
    # Create test files with different content
    file_manager = file_manager
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
            records=[
                {"content_type": "document", "content_text": content},
                {"content_type": "paragraph", "content_text": content},
            ],
            full_text=content,
            metadata={"file_size": len(content), "file_type": filename.split(".")[-1]},
        )

    # Search for chocolate cookies related content
    query = "olympic medals"
    results = file_manager._search_files(references={"full_text": query}, k=3)

    assert len(results) >= 1
    assert isinstance(results[0], File)
    # Should find sports report first
    assert results[0].filename == "sports_report.docx"

    # Verify columns were created
    cols = file_manager._list_columns()
    assert "_full_text_emb" in cols


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_search_files_multi_columns(file_manager):
    """Test semantic search across multiple columns."""
    file_manager = file_manager
    file_manager._create_file(
        filename="wildlife_guide.pdf",
        status="success",
        records=[
            {
                "content_type": "document",
                "content_text": "Information about African elephants and their migratory patterns.",
            },
            {
                "content_type": "paragraph",
                "content_text": "African elephants migrate across vast distances.",
            },
        ],
        full_text="Information about African elephants and their migratory patterns.",
        metadata={"file_type": "pdf", "category": "nature reference"},
        description="Information about African elephants and their migratory patterns.",
    )

    file_manager._create_file(
        filename="theatre_review.docx",
        status="success",
        records=[
            {
                "content_type": "document",
                "content_text": "Critical analysis of Shakespeare's Hamlet performances.",
            },
            {
                "content_type": "paragraph",
                "content_text": "Hamlet's soliloquies are central to the play.",
            },
        ],
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
def test_search_files_ranking_precision_k1(file_manager):
    """Test that search correctly ranks documents and returns most relevant first (k=1)."""
    # Create multiple documents with different relevance levels for the same query
    file_manager = file_manager
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
            records=[
                {"content_type": "document", "content_text": content},
                {"content_type": "paragraph", "content_text": content},
            ],
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
def test_search_files_ranking_precision_k3(file_manager):
    """Test search ranking with k=3 to verify correct ordering."""
    # Create documents with varying degrees of relevance
    file_manager = file_manager
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
            records=[
                {"content_type": "document", "content_text": content},
                {"content_type": "paragraph", "content_text": content},
            ],
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
def test_search_files_exact_match_beats_partial(file_manager):
    """Test that exact keyword matches rank higher than partial matches."""
    file_manager = file_manager
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
            records=[
                {"content_type": "document", "content_text": content},
                {"content_type": "paragraph", "content_text": content},
            ],
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
def test_search_files_multiple_reference_columns(file_manager):
    """Test search across multiple reference columns with correct ranking."""
    # Create files where different signals appear in different fields
    file_manager = file_manager
    file_manager._create_file(
        filename="signal_in_text.pdf",
        status="success",
        records=[
            {
                "content_type": "document",
                "content_text": "Deep learning neural networks for computer vision and natural language processing.",
            },
            {
                "content_type": "paragraph",
                "content_text": "Computer vision applications use neural networks.",
            },
        ],
        full_text="Deep learning neural networks for computer vision and natural language processing.",
        metadata={"category": "general", "keywords": "basic"},
        description="Deep learning neural networks for computer vision and natural language processing.",
    )

    file_manager._create_file(
        filename="signal_in_both.docx",
        status="success",
        records=[
            {
                "content_type": "document",
                "content_text": "Advanced machine learning techniques including neural networks and deep learning algorithms.",
            },
            {
                "content_type": "paragraph",
                "content_text": "Neural networks and deep learning are key techniques.",
            },
        ],
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
        records=[
            {
                "content_type": "document",
                "content_text": "This is a general technology document.",
            },
            {
                "content_type": "paragraph",
                "content_text": "General technology overview.",
            },
        ],
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
def test_search_files_domain_specific_ranking(file_manager):
    """Test search ranking for domain-specific queries."""
    # Create documents in different domains
    file_manager = file_manager
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
            records=[
                {"content_type": "document", "content_text": content},
                {"content_type": "paragraph", "content_text": content},
            ],
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
def test_filter_files_basic(file_manager):
    """Test basic filtering of files."""
    # Create test files
    file_manager = file_manager
    file_manager._create_file(
        filename="document.pdf",
        status="success",
        records=[
            {"content_type": "document", "content_text": "PDF content"},
            {"content_type": "paragraph", "content_text": "This is a PDF document."},
        ],
        full_text="PDF content",
        metadata={"file_size": 2048},
    )

    file_manager._create_file(
        filename="spreadsheet.xlsx",
        status="error",
        error="Parse failed",
        records=[],
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
def test_filter_files_metadata(file_manager):
    """Test filtering files by metadata fields."""
    file_manager = file_manager
    file_manager._create_file(
        filename="large_file.pdf",
        status="success",
        records=[
            {"content_type": "document", "content_text": "Large document content"},
            {
                "content_type": "paragraph",
                "content_text": "This is a large document with lots of content.",
            },
        ],
        full_text="Large document content",
        metadata={"file_size": 5000000, "file_type": "pdf"},
    )

    file_manager._create_file(
        filename="small_file.txt",
        status="success",
        records=[
            {"content_type": "document", "content_text": "Small text content"},
            {"content_type": "paragraph", "content_text": "This is a small text file."},
        ],
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
def test_search_files_no_results_backfill(file_manager):
    """Test that search falls back to recent files when no semantic matches."""
    file_manager = file_manager
    file_manager._create_file(
        filename="random_doc.txt",
        status="success",
        records=[
            {
                "content_type": "document",
                "content_text": "Random content about nothing relevant",
            },
            {
                "content_type": "paragraph",
                "content_text": "This is random unrelated content.",
            },
        ],
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
def test_list_columns(file_manager):
    """Test _list_columns functionality."""
    # Create a file to ensure table exists
    file_manager = file_manager
    file_manager._create_file(
        filename="test.txt",
        status="success",
        records=[
            {"content_type": "document", "content_text": "test content"},
            {"content_type": "paragraph", "content_text": "This is test content."},
        ],
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
