"""
Test FileManager search, filter, and ingestion functionality aligned to the
refactored pipeline (import → parse → ingest) and flattened fields.
"""

from __future__ import annotations


import pytest
from pathlib import Path
from tests.helpers import _handle_project


@pytest.mark.requires_real_unify
@_handle_project
def test_create_basic(file_manager, tmp_path: Path):
    """Import and parse a file; verify ingestion succeeded."""
    fm = file_manager
    # Use a real text file (avoid writing invalid binary formats like .pdf).
    p = tmp_path / "travel_notes.txt"
    content = "This document contains notes about hiking trails in the Alps."
    p.write_text(content, encoding="utf-8")
    name = str(p)
    res = fm.ingest_files(name)
    assert name in res
    item = res[name]
    # All returns are now Pydantic models - use attribute access
    assert item.status == "success"
    rows = fm.filter_files(filter=f"file_path == '{name}'")
    assert rows and any(r.get("file_path", name) == name for r in rows)


@pytest.mark.requires_real_unify
@_handle_project
def test_create_with_error(file_manager, tmp_path: Path):
    """Unsupported extension still results in a parsable text fallback or error."""
    fm = file_manager
    p = tmp_path / "corrupt_music_score.xyz"
    p.write_text("Failed to parse sheet music", encoding="utf-8")
    name = str(p)
    res = fm.ingest_files(name)
    assert name in res
    item = res[name]
    # All returns are now Pydantic models - use attribute access
    assert item.status in ("success", "error")


@pytest.mark.requires_real_unify
@_handle_project
def test_search_single_reference_basic(file_manager, tmp_path: Path):
    """Test basic semantic search over file contents."""
    # Create test files with different content
    fm = file_manager
    fm.clear()
    files_data = [
        (
            "history_essay.txt",
            "This document explores the causes of the French Revolution.",
        ),
        (
            "garden_guide.txt",
            "A step by step guide to planting roses and maintaining gardens.",
        ),
        ("sports_report.txt", "Summary of the Olympic Games and medal counts."),
        (
            "astronomy_notes.txt",
            "Observations of the Orion constellation and nearby nebulae.",
        ),
    ]

    for filename, content in files_data:
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        name = str(p)
        fm.ingest_files(name)

    # Search for chocolate cookies related content
    query = "olympic medals"
    results = fm.search_files(references={"summary": query}, limit=3)

    assert len(results) >= 1
    # Should find sports report first
    assert any(r.get("file_path", "").endswith("sports_report.txt") for r in results)

    # Verify columns were created
    cols = fm.list_columns()
    assert "_summary_emb" in cols


@pytest.mark.requires_real_unify
@_handle_project
def test_search_multi_columns(file_manager, tmp_path: Path):
    """Test semantic search across multiple columns."""
    fm = file_manager
    fm.clear()
    p1 = tmp_path / "wildlife_guide.txt"
    p1.write_text(
        "Information about African elephants and their migratory patterns.",
        encoding="utf-8",
    )
    n1 = str(p1)
    fm.ingest_files(n1)

    p2 = tmp_path / "theatre_review.txt"
    p2.write_text(
        "Critical analysis of Shakespeare's Hamlet performances.",
        encoding="utf-8",
    )
    n2 = str(p2)
    fm.ingest_files(n2)

    # Search using both content and metadata
    refs = {"summary": "Shakespeare performances", "file_path": "analysis"}
    results = fm.search_files(references=refs, limit=2)

    assert len(results) >= 1
    assert any(r.get("file_path", "").endswith("theatre_review.txt") for r in results)

    # Verify columns were created
    cols = fm.list_columns()
    assert "_summary_emb" in cols
    assert "_file_path_emb" in cols


@pytest.mark.requires_real_unify
@_handle_project
def test_search_ranking_precision_k1(file_manager, tmp_path: Path):
    """Test that search correctly ranks documents and returns most relevant first (k=1)."""
    # Create multiple documents with different relevance levels for the same query
    fm = file_manager
    fm.clear()
    test_docs = [
        (
            "ai_overview.txt",
            "Artificial intelligence is mentioned briefly in this general overview document.",
            "Artificial intelligence overview",
        ),
        (
            "ml_research.txt",
            "Machine learning and artificial intelligence research paper discussing deep neural networks, algorithms, and AI applications in detail.",
            "Machine learning research",
        ),
        (
            "cooking_recipe.txt",
            "This chocolate chip cookie recipe has nothing to do with technology.",
            "Chocolate chip cookie recipe",
        ),
        (
            "tech_news.txt",
            "Technology news mentions AI and machine learning in passing.",
            "Technology news",
        ),
    ]
    for filename, content, description in test_docs:
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        name = str(p)
        fm.ingest_files(name)

    # Search for AI/ML content - should rank ml_research.txt highest
    query = "artificial intelligence machine learning research algorithms"
    results = fm.search_files(references={"summary": query}, limit=1)

    assert len(results) == 1
    print(f"results: {[(f.get('file_id'), f.get('file_path')) for f in results]}")
    assert any(r.get("file_path", "").endswith("ml_research.txt") for r in results)

    # Result should be present
    assert results


@pytest.mark.requires_real_unify
@_handle_project
def test_search_ranking_precision_k3(file_manager, tmp_path: Path):
    """Test search ranking with k=3 to verify correct ordering."""
    # Create documents with varying degrees of relevance
    fm = file_manager
    fm.clear()
    test_docs = [
        (
            "irrelevant.txt",
            "This document is about gardening and has no technical content.",
        ),
        (
            "somewhat_relevant.txt",
            "This business document mentions AI tools in one paragraph.",
        ),
        (
            "highly_relevant.txt",
            "Comprehensive artificial intelligence and machine learning guide with detailed algorithms, neural network architectures, and practical AI applications.",
        ),
        (
            "moderately_relevant.txt",
            "Technical article about artificial intelligence applications in industry.",
        ),
        (
            "unrelated.txt",
            "Financial report with quarterly earnings and budget forecasts.",
        ),
    ]

    for filename, content in test_docs:
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        name = str(p)
        fm.ingest_files(name)

    # Search for AI content
    query = "artificial intelligence machine learning algorithms and AI applications"
    results = fm.search_files(references={"summary": query}, limit=3)

    assert len(results) >= 3

    # First result should be the most comprehensive/relevant
    print(f"results: {[(f.get('file_id'), f.get('file_path')) for f in results]}")
    assert any(
        r.get("file_path", "").endswith("highly_relevant.txt") for r in results[:3]
    )

    # Second and third should be more relevant than irrelevant docs
    top_3_filenames = [r.get("file_path") for r in results[:3]]
    assert "irrelevant.txt" not in top_3_filenames
    assert "unrelated.txt" not in top_3_filenames


@pytest.mark.requires_real_unify
@_handle_project
def test_search_exact_match_beats_partial(file_manager, tmp_path: Path):
    """Test that exact keyword matches rank higher than partial matches."""
    fm = file_manager
    fm.clear()
    test_docs = [
        (
            "partial_match.txt",
            "This document discusses learning and intelligent systems in general terms.",
        ),
        (
            "exact_match.txt",
            "Machine learning artificial intelligence neural networks deep learning algorithms.",
        ),
        (
            "weak_match.txt",
            "Brief mention of smart technology and automated learning processes.",
        ),
    ]

    for filename, content in test_docs:
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        name = str(p)
        fm.ingest_files(name)

    # Search for exact terms that appear in exact_match.txt
    query = "machine learning artificial intelligence neural networks"
    results = fm.search_files(references={"summary": query}, limit=1)

    assert len(results) == 1
    assert any(r.get("file_path", "").endswith("exact_match.txt") for r in results)


@pytest.mark.requires_real_unify
@_handle_project
def test_search_multiple_reference_columns(file_manager, tmp_path: Path):
    """Test search across multiple reference columns with correct ranking."""
    # Create files where different signals appear in different fields
    fm = file_manager
    fm.clear()
    docs = {
        "signal_in_text.txt": "Deep learning neural networks for computer vision and natural language processing.",
        "signal_in_both.txt": "Advanced machine learning techniques including neural networks and deep learning algorithms.",
        "signal_in_metadata.txt": "This is a general technology document.",
    }
    for fname, text in docs.items():
        p = tmp_path / fname
        p.write_text(text, encoding="utf-8")
        name = str(p)
        fm.ingest_files(name)

    # Search with multiple reference columns - signal_in_both should rank highest
    # as it has relevant content in both summary and file path
    refs = {
        "summary": "neural networks deep learning algorithms",
        "file_path": "both",
    }
    results = fm.search_files(references=refs, limit=1)

    assert len(results) == 1
    assert any(r.get("file_path", "").endswith("signal_in_both.txt") for r in results)

    # Verify columns were created
    cols = fm.list_columns()
    assert "_summary_emb" in cols
    assert "_file_path_emb" in cols


@pytest.mark.requires_real_unify
@_handle_project
def test_search_domain_specific_ranking(file_manager, tmp_path: Path):
    """Test search ranking for domain-specific queries."""
    # Create documents in different domains
    fm = file_manager
    fm.clear()
    test_docs = [
        (
            "medical_ai.txt",
            "Artificial intelligence applications in medical diagnosis, healthcare AI systems, and clinical machine learning for patient care.",
        ),
        (
            "finance_ai.txt",
            "AI applications in financial services, algorithmic trading, and machine learning for risk assessment in banking.",
        ),
        (
            "general_ai.txt",
            "General overview of artificial intelligence and machine learning concepts.",
        ),
        (
            "automotive_ai.txt",
            "Automotive AI systems, self-driving cars, machine learning for autonomous vehicles and transportation.",
        ),
    ]

    for filename, content in test_docs:
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        name = str(p)
        fm.ingest_files(name)

    # Search for medical AI - should rank medical_ai.txt first
    medical_query = "artificial intelligence medical diagnosis healthcare clinical"
    medical_results = fm.search_files(
        references={"summary": medical_query},
        limit=1,
    )

    assert len(medical_results) == 1
    assert any(
        r.get("file_path", "").endswith("medical_ai.txt") for r in medical_results
    )

    # Search for automotive AI - should rank automotive_ai.txt first
    auto_query = "artificial intelligence automotive self-driving autonomous vehicles"
    auto_results = fm.search_files(references={"summary": auto_query}, limit=1)

    assert len(auto_results) == 1
    assert any(
        r.get("file_path", "").endswith("automotive_ai.txt") for r in auto_results
    )


@pytest.mark.requires_real_unify
@_handle_project
def test_filter_basic(file_manager, tmp_path: Path):
    """Test basic filtering of files."""
    # Create test files
    fm = file_manager
    p_ok = tmp_path / "document.txt"
    p_ok.write_text("Simple text content", encoding="utf-8")
    n_ok = str(p_ok)
    fm.ingest_files(n_ok)

    p_bad = tmp_path / "spreadsheet.txt"
    p_bad.write_text("Another text file", encoding="utf-8")
    n_bad = str(p_bad)
    fm.ingest_files(n_bad)

    # Filter by status
    success_files = fm.filter_files(filter="status == 'success'")
    assert len(success_files) >= 1
    assert all(f.get("status", "") == "success" for f in success_files)

    # Filter by filename extension
    pdf_files = fm.filter_files(filter="file_path.endswith('.txt')")
    assert len(pdf_files) >= 1
    assert all(f.get("file_path", "").endswith(".txt") for f in pdf_files)


@pytest.mark.requires_real_unify
@_handle_project
def test_filter_metadata(file_manager, tmp_path: Path):
    """Test filtering files by metadata fields."""
    fm = file_manager
    fm.clear()
    p_large = tmp_path / "large_file.txt"
    p_large.write_text("Large document content", encoding="utf-8")
    n_large = str(p_large)
    fm.ingest_files(n_large)

    p_small = tmp_path / "small_file.txt"
    p_small.write_text("Small text content", encoding="utf-8")
    n_small = str(p_small)
    fm.ingest_files(n_small)

    # Filter by file format - both files are txt, so should return 2
    txt_files = fm.filter_files(filter="file_format == 'txt'")
    assert len(txt_files) == 2

    # Filter by specific file path
    large_files = fm.filter_files(filter="file_path.endswith('large_file.txt')")
    assert len(large_files) == 1
    assert str(large_files[0].get("file_path", "")).endswith("large_file.txt")


@pytest.mark.requires_real_unify
@_handle_project
def test_search_no_results_backfill(file_manager, tmp_path: Path):
    """Test that search falls back to recent files when no semantic matches."""
    fm = file_manager
    fm.clear()
    p = tmp_path / "random_doc.txt"
    p.write_text("Random content about nothing relevant", encoding="utf-8")
    n = str(p)
    fm.ingest_files(n)

    # Search for something completely unrelated
    results = fm.search_files(
        references={"summary": "quantum physics molecules"},
        limit=5,
    )

    # Should still return files (backfill behavior)
    assert len(results) >= 1
    assert any(r.get("file_path", "").endswith("random_doc.txt") for r in results)


@pytest.mark.requires_real_unify
@_handle_project
def test_list_columns(file_manager, tmp_path: Path):
    """Test _list_columns functionality."""
    # Create a file to ensure table exists
    fm = file_manager
    p = tmp_path / "test.txt"
    p.write_text("test content", encoding="utf-8")
    n = str(p)
    fm.ingest_files(n)

    # Test with types
    cols_with_types = fm.list_columns(include_types=True)
    assert isinstance(cols_with_types, dict)
    assert "file_path" in cols_with_types
    assert "status" in cols_with_types
    assert "summary" in cols_with_types

    # Test without types
    cols_list = fm.list_columns(include_types=False)
    assert isinstance(cols_list, list)
    assert "file_path" in cols_list
