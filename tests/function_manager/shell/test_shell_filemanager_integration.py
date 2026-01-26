"""
Realistic integration test: Shell scripts combining lexical and semantic search.

This test demonstrates the "best of both worlds" approach where shell scripts
combine:
- Lexical operations (grep, find, wc, awk, etc.) for pattern matching
- Semantic search via FileManager primitives for meaning-based retrieval

The test sets up a realistic filesystem with .txt and .csv files, ingests them
into FileManager, then executes a shell script that:
1. Uses grep/find for fast lexical pattern matching
2. Uses FileManager.search_files for semantic similarity search
3. Combines both approaches for comprehensive file analysis
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.helpers import _handle_project
from unity.function_manager.function_manager import FunctionManager
from unity.common.context_registry import ContextRegistry

# ────────────────────────────────────────────────────────────────────────────
# Sample Files for Realistic Test
# ────────────────────────────────────────────────────────────────────────────

# Text files representing various documents
INVOICE_2024_TXT = """
Invoice #INV-2024-001
Date: 2024-01-15
Customer: Acme Corporation
Contact: alice@acme.com

Items:
- Widget A x 100 @ $10.00 = $1,000.00
- Widget B x 50 @ $25.00 = $1,250.00
- Consulting services (10 hours) @ $150/hr = $1,500.00

Subtotal: $3,750.00
Tax (8%): $300.00
Total: $4,050.00

Payment Terms: Net 30
Due Date: 2024-02-14

Notes: Priority customer - expedited shipping included
"""

INVOICE_2023_TXT = """
Invoice #INV-2023-156
Date: 2023-11-20
Customer: TechStart Inc.
Contact: bob@techstart.io

Items:
- Enterprise License x 1 @ $5,000.00 = $5,000.00
- Support Package (Annual) @ $1,200.00 = $1,200.00

Subtotal: $6,200.00
Tax (8%): $496.00
Total: $6,696.00

Payment Terms: Net 45
Due Date: 2024-01-04

Notes: Renewal from 2022 contract
"""

PROJECT_NOTES_TXT = """
Project: Customer Analytics Dashboard
Last Updated: 2024-03-01
Status: In Progress

Team Members:
- Alice (Lead Developer)
- Bob (Data Engineer)
- Carol (UX Designer)

Key Milestones:
1. Data pipeline setup - COMPLETED
2. Dashboard wireframes - COMPLETED
3. Backend API development - IN PROGRESS
4. Frontend implementation - PENDING
5. User testing - PENDING

Budget Status:
- Allocated: $50,000
- Spent: $32,000
- Remaining: $18,000

Risks:
- API integration delays due to vendor response times
- Possible scope creep from additional reporting requests
"""

MEETING_NOTES_TXT = """
Meeting: Q1 Budget Review
Date: 2024-01-10
Attendees: Alice, Bob, Carol, Dave (Finance)

Agenda:
1. Review Q4 2023 spending
2. Approve Q1 2024 budget allocations
3. Discuss cost optimization opportunities

Key Decisions:
- Approved $150,000 for software licenses
- Deferred hardware upgrades to Q2
- Allocated $30,000 for training programs

Action Items:
- Alice: Submit vendor quotes by Jan 15
- Bob: Prepare cost analysis report
- Carol: Schedule follow-up with IT department

Next Meeting: 2024-02-07
"""

# CSV files representing structured data
SALES_DATA_CSV = """date,product,quantity,unit_price,total,customer,region
2024-01-01,Widget A,50,10.00,500.00,Acme Corp,North
2024-01-02,Widget B,25,25.00,625.00,TechStart,West
2024-01-03,Widget A,100,10.00,1000.00,GlobalTech,East
2024-01-04,Widget C,15,50.00,750.00,Acme Corp,North
2024-01-05,Widget B,30,25.00,750.00,DataCo,South
2024-01-06,Widget A,75,10.00,750.00,TechStart,West
2024-01-07,Widget C,20,50.00,1000.00,GlobalTech,East
2024-01-08,Widget B,40,25.00,1000.00,Acme Corp,North
"""

CUSTOMER_DATA_CSV = """customer_id,name,email,region,tier,annual_revenue
1,Acme Corp,alice@acme.com,North,Enterprise,500000
2,TechStart,bob@techstart.io,West,Startup,75000
3,GlobalTech,carol@globaltech.com,East,Enterprise,1200000
4,DataCo,dave@dataco.net,South,SMB,150000
5,InnovateLabs,eve@innovatelabs.org,West,Startup,50000
"""

EXPENSES_CSV = """date,category,amount,department,description,approved_by
2024-01-05,Software,1500.00,Engineering,Annual IDE licenses,Alice
2024-01-08,Travel,800.00,Sales,Client meeting - NYC,Bob
2024-01-10,Equipment,2500.00,Engineering,Development laptops,Alice
2024-01-12,Training,600.00,HR,Leadership workshop,Carol
2024-01-15,Software,3000.00,Engineering,Cloud hosting Q1,Alice
2024-01-18,Marketing,1200.00,Marketing,Trade show booth,Dave
"""


# ────────────────────────────────────────────────────────────────────────────
# Shell Script: Combining Lexical and Semantic Search
# ────────────────────────────────────────────────────────────────────────────

HYBRID_SEARCH_SCRIPT = """#!/bin/sh
# analyze_documents.sh
#
# This script combines lexical (grep/find) and semantic (FileManager) search
# to analyze a collection of documents.
#
# Usage: Called by FunctionManager.execute_shell_script with primitives

echo "=== Document Analysis Report ==="
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Lexical Search (grep/find/awk)
# Fast pattern matching for known keywords
# ─────────────────────────────────────────────────────────────────────────────

echo "--- LEXICAL ANALYSIS ---"
echo ""

# Find all text files
txt_count=$(find "$DATA_DIR" -name "*.txt" | wc -l | tr -d ' ')
echo "Text files found: $txt_count"

# Find all CSV files
csv_count=$(find "$DATA_DIR" -name "*.csv" | wc -l | tr -d ' ')
echo "CSV files found: $csv_count"

# Search for invoices containing specific amounts
echo ""
echo "Invoices over $5000:"
grep -l "Total:.*\\$[5-9][0-9][0-9][0-9]\\|Total:.*\\$[1-9][0-9][0-9][0-9][0-9]" "$DATA_DIR"/*.txt 2>/dev/null || echo "  None found"

# Find files mentioning specific people
echo ""
echo "Files mentioning Alice:"
grep -l "Alice" "$DATA_DIR"/* 2>/dev/null | while read f; do
    echo "  - $(basename "$f")"
done

# Count total lines in all CSV files
echo ""
csv_lines=$(cat "$DATA_DIR"/*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "Total CSV data rows: $csv_lines"

# Extract unique customers from sales data
echo ""
echo "Unique customers in sales data:"
if [ -f "$DATA_DIR/sales_data.csv" ]; then
    tail -n +2 "$DATA_DIR/sales_data.csv" | cut -d',' -f6 | sort -u | while read cust; do
        echo "  - $cust"
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Semantic Search (FileManager primitives)
# Meaning-based retrieval using embeddings
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "--- SEMANTIC ANALYSIS ---"
echo ""

# Search for documents related to "budget" concepts
echo "Documents semantically related to 'budget planning':"
budget_results=$(unity-primitive files search_files --references '{"query": "budget planning and financial allocation"}' --k 3)
echo "$budget_results" | head -20

# Search for documents about customer relationships
echo ""
echo "Documents about 'customer relationship management':"
crm_results=$(unity-primitive files search_files --references '{"query": "customer contacts and enterprise accounts"}' --k 3)
echo "$crm_results" | head -20

# Use filter to find high-value entries
echo ""
echo "High-value transactions (using filter):"
filter_results=$(unity-primitive files filter_files --filter "total > 1000" --limit 5 --tables '["sales_data"]')
echo "$filter_results" | head -20

# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Combined Analysis
# Leveraging both approaches for comprehensive results
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "--- COMBINED ANALYSIS ---"
echo ""

# First, use grep to find files with "Priority" keyword
priority_files=$(grep -l "Priority\\|priority" "$DATA_DIR"/*.txt 2>/dev/null)
if [ -n "$priority_files" ]; then
    echo "Priority documents (lexical match):"
    for f in $priority_files; do
        echo "  - $(basename "$f")"
    done

    # Then use semantic search to find related documents
    echo ""
    echo "Related documents via semantic search:"
    related=$(unity-primitive files search_files --references '{"query": "urgent priority high importance"}' --k 2)
    echo "$related" | head -10
fi

# Summary statistics combining both methods
echo ""
echo "=== SUMMARY ==="
echo "Files analyzed: $((txt_count + csv_count))"
echo "Lexical matches (Alice): $(grep -c "Alice" "$DATA_DIR"/* 2>/dev/null | grep -v ":0$" | wc -l | tr -d ' ') files"

echo ""
echo "Analysis complete."
"""


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory():
    """Factory fixture that creates FunctionManager instances."""
    managers = []

    def _create():
        ContextRegistry.forget(FunctionManager, "Functions/VirtualEnvs")
        ContextRegistry.forget(FunctionManager, "Functions/Compositional")
        ContextRegistry.forget(FunctionManager, "Functions/Primitives")
        ContextRegistry.forget(FunctionManager, "Functions/Meta")
        fm = FunctionManager()
        managers.append(fm)
        return fm

    yield _create

    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


@pytest.fixture
def sample_data_dir():
    """Create a temporary directory with sample data files."""
    with tempfile.TemporaryDirectory(prefix="test_data_") as tmpdir:
        data_dir = Path(tmpdir)

        # Create text files
        (data_dir / "invoice_2024.txt").write_text(INVOICE_2024_TXT)
        (data_dir / "invoice_2023.txt").write_text(INVOICE_2023_TXT)
        (data_dir / "project_notes.txt").write_text(PROJECT_NOTES_TXT)
        (data_dir / "meeting_notes.txt").write_text(MEETING_NOTES_TXT)

        # Create CSV files
        (data_dir / "sales_data.csv").write_text(SALES_DATA_CSV)
        (data_dir / "customer_data.csv").write_text(CUSTOMER_DATA_CSV)
        (data_dir / "expenses.csv").write_text(EXPENSES_CSV)

        yield str(data_dir)


@pytest.fixture
def mock_file_primitives():
    """
    Create a mock primitives object that simulates FileManager behavior.

    In a real integration test, this would be replaced with actual FileManager
    that has ingested the sample files. For this test, we simulate realistic
    responses.
    """
    primitives = MagicMock()
    primitives.files = MagicMock()

    # Simulate search_files returning semantically relevant results
    async def mock_search_files(references=None, k=10, **kwargs):
        query = references.get("query", "") if references else ""
        results = []

        if "budget" in query.lower() or "financial" in query.lower():
            results = [
                {
                    "file_path": "/data/meeting_notes.txt",
                    "snippet": "Q1 Budget Review... Approved $150,000",
                    "score": 0.92,
                },
                {
                    "file_path": "/data/project_notes.txt",
                    "snippet": "Budget Status: Allocated $50,000",
                    "score": 0.85,
                },
                {
                    "file_path": "/data/expenses.csv",
                    "snippet": "Engineering expenses breakdown",
                    "score": 0.78,
                },
            ]
        elif "customer" in query.lower() or "enterprise" in query.lower():
            results = [
                {
                    "file_path": "/data/customer_data.csv",
                    "snippet": "Enterprise tier customers with revenue",
                    "score": 0.94,
                },
                {
                    "file_path": "/data/invoice_2024.txt",
                    "snippet": "Customer: Acme Corporation",
                    "score": 0.82,
                },
            ]
        elif "priority" in query.lower() or "urgent" in query.lower():
            results = [
                {
                    "file_path": "/data/invoice_2024.txt",
                    "snippet": "Priority customer - expedited shipping",
                    "score": 0.88,
                },
                {
                    "file_path": "/data/project_notes.txt",
                    "snippet": "Key risks and priority items",
                    "score": 0.75,
                },
            ]

        return results[:k]

    # Simulate filter_files for structured queries
    async def mock_filter_files(filter=None, limit=100, tables=None, **kwargs):
        if "total > 1000" in str(filter):
            return [
                {
                    "date": "2024-01-03",
                    "product": "Widget A",
                    "total": 1000.00,
                    "customer": "GlobalTech",
                },
                {
                    "date": "2024-01-07",
                    "product": "Widget C",
                    "total": 1000.00,
                    "customer": "GlobalTech",
                },
                {
                    "date": "2024-01-08",
                    "product": "Widget B",
                    "total": 1000.00,
                    "customer": "Acme Corp",
                },
            ]
        return []

    primitives.files.search_files = AsyncMock(side_effect=mock_search_files)
    primitives.files.filter_files = AsyncMock(side_effect=mock_filter_files)

    return primitives


# ────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
@pytest.mark.asyncio
async def test_hybrid_lexical_semantic_search(
    function_manager_factory,
    sample_data_dir,
    mock_file_primitives,
):
    """
    Test the hybrid approach combining lexical shell operations with semantic search.

    This test demonstrates:
    1. Shell scripts can access local filesystem for grep/find operations
    2. Shell scripts can call FileManager primitives for semantic search
    3. Both approaches work together for comprehensive document analysis
    """
    fm = function_manager_factory()

    result = await fm.execute_shell_script(
        implementation=HYBRID_SEARCH_SCRIPT,
        language="sh",
        env={"DATA_DIR": sample_data_dir},
        primitives=mock_file_primitives,
        timeout=30.0,
    )

    # Check execution succeeded
    assert (
        result["error"] is None
    ), f"Script failed: {result['stderr']}\nStdout: {result['stdout']}"

    stdout = result["stdout"]

    # Verify lexical analysis results
    assert "Text files found: 4" in stdout
    assert "CSV files found: 3" in stdout
    assert "Files mentioning Alice" in stdout
    assert "Unique customers in sales data" in stdout
    assert "Acme Corp" in stdout

    # Verify semantic analysis was called
    mock_file_primitives.files.search_files.assert_called()
    mock_file_primitives.files.filter_files.assert_called()

    # Verify combined analysis section
    assert "COMBINED ANALYSIS" in stdout
    assert "Analysis complete" in stdout


@_handle_project
@pytest.mark.asyncio
async def test_lexical_search_only(function_manager_factory, sample_data_dir):
    """Test pure lexical search without primitives."""
    fm = function_manager_factory()

    script = """#!/bin/sh
# Pure lexical analysis
echo "=== Lexical Search Only ==="

# Count files by type
echo "TXT files: $(find "$DATA_DIR" -name "*.txt" | wc -l | tr -d ' ')"
echo "CSV files: $(find "$DATA_DIR" -name "*.csv" | wc -l | tr -d ' ')"

# Find invoices
echo ""
echo "Invoice files:"
ls "$DATA_DIR"/*invoice* 2>/dev/null | while read f; do
    echo "  - $(basename "$f")"
done

# Extract totals from invoices using regex
echo ""
echo "Invoice totals:"
grep "^Total:" "$DATA_DIR"/*.txt 2>/dev/null | while read line; do
    echo "  $line"
done

# Analyze CSV structure
echo ""
echo "CSV columns in sales_data.csv:"
head -1 "$DATA_DIR/sales_data.csv" | tr ',' '\n' | while read col; do
    echo "  - $col"
done
"""

    result = await fm.execute_shell_script(
        implementation=script,
        language="sh",
        env={"DATA_DIR": sample_data_dir},
    )

    assert result["error"] is None, f"Script failed: {result['stderr']}"

    stdout = result["stdout"]
    assert "TXT files: 4" in stdout
    assert "CSV files: 3" in stdout
    assert "invoice_2024.txt" in stdout or "invoice_2023.txt" in stdout
    assert "Total:" in stdout
    assert "date" in stdout  # First CSV column


@_handle_project
@pytest.mark.asyncio
async def test_semantic_search_only(
    function_manager_factory,
    mock_file_primitives,
):
    """Test pure semantic search via primitives."""
    fm = function_manager_factory()

    script = """#!/bin/sh
# Pure semantic analysis via FileManager
echo "=== Semantic Search Only ==="

# Search for budget-related documents
echo "Budget-related documents:"
unity-primitive files search_files --references '{"query": "budget financial planning"}' --k 5

# Search for customer information
echo ""
echo "Customer-related documents:"
unity-primitive files search_files --references '{"query": "customer enterprise accounts"}' --k 3

# Filter for high-value transactions
echo ""
echo "High-value transactions:"
unity-primitive files filter_files --filter "total > 1000" --limit 10 --tables '["sales_data"]'
"""

    result = await fm.execute_shell_script(
        implementation=script,
        language="sh",
        primitives=mock_file_primitives,
    )

    assert result["error"] is None, f"Script failed: {result['stderr']}"

    # Verify all semantic methods were called
    assert mock_file_primitives.files.search_files.call_count >= 2
    assert mock_file_primitives.files.filter_files.call_count >= 1

    stdout = result["stdout"]
    assert "Budget-related documents" in stdout
    assert "Customer-related documents" in stdout


@_handle_project
@pytest.mark.asyncio
async def test_data_extraction_pipeline(
    function_manager_factory,
    sample_data_dir,
    mock_file_primitives,
):
    """
    Test a realistic data extraction pipeline that combines shell text processing
    with semantic enrichment.
    """
    fm = function_manager_factory()

    script = """#!/bin/sh
# Data extraction pipeline combining lexical and semantic processing

echo "=== Data Extraction Pipeline ==="

# Step 1: Use awk to aggregate sales by customer (lexical)
echo ""
echo "Step 1: Sales aggregation (awk)"
if [ -f "$DATA_DIR/sales_data.csv" ]; then
    echo "Sales by customer:"
    tail -n +2 "$DATA_DIR/sales_data.csv" | awk -F',' '
        { sales[$6] += $5 }
        END { for (c in sales) printf "  %s: $%.2f\\n", c, sales[c] }
    '
fi

# Step 2: Extract expense categories (lexical)
echo ""
echo "Step 2: Expense categories (cut/sort)"
if [ -f "$DATA_DIR/expenses.csv" ]; then
    echo "Expense breakdown:"
    tail -n +2 "$DATA_DIR/expenses.csv" | awk -F',' '
        { exp[$2] += $3 }
        END { for (c in exp) printf "  %s: $%.2f\\n", c, exp[c] }
    '
fi

# Step 3: Semantic search for context (semantic)
echo ""
echo "Step 3: Semantic context for top expenses"
unity-primitive files search_files --references '{"query": "software licensing costs and engineering budget"}' --k 2

# Step 4: Combine results
echo ""
echo "=== Pipeline Complete ==="
echo "Lexical processing: CSV aggregation with awk"
echo "Semantic enrichment: Context from related documents"
"""

    result = await fm.execute_shell_script(
        implementation=script,
        language="sh",
        env={"DATA_DIR": sample_data_dir},
        primitives=mock_file_primitives,
    )

    assert result["error"] is None, f"Script failed: {result['stderr']}"

    stdout = result["stdout"]

    # Verify lexical processing results
    assert "Sales by customer" in stdout
    assert "Expense breakdown" in stdout

    # Verify semantic enrichment was called
    assert mock_file_primitives.files.search_files.called

    # Verify pipeline completed
    assert "Pipeline Complete" in stdout


@_handle_project
@pytest.mark.asyncio
async def test_error_resilience(
    function_manager_factory,
    sample_data_dir,
):
    """
    Test that scripts handle errors gracefully when primitives fail.
    """
    fm = function_manager_factory()

    # Create mock that fails
    mock_p = MagicMock()
    mock_p.files = MagicMock()
    mock_p.files.search_files = AsyncMock(
        side_effect=RuntimeError("FileManager not initialized"),
    )

    script = """#!/bin/sh
# Error-resilient script

echo "=== Running with error handling ==="

# Lexical search always works
echo "Lexical search:"
file_count=$(find "$DATA_DIR" -type f | wc -l | tr -d ' ')
echo "  Files found: $file_count"

# Semantic search may fail
echo ""
echo "Semantic search (may fail):"
semantic_result=$(unity-primitive files search_files --references '{"query": "test"}' --k 1 2>&1)
if [ $? -eq 0 ]; then
    echo "  Success: $semantic_result"
else
    echo "  Failed (graceful fallback): Using lexical results only"
fi

echo ""
echo "Script completed with fallback handling"
"""

    result = await fm.execute_shell_script(
        implementation=script,
        language="sh",
        env={"DATA_DIR": sample_data_dir},
        primitives=mock_p,
    )

    # Script should complete even if semantic search fails
    assert result["result"] == 0, f"Script failed: {result['stderr']}"

    stdout = result["stdout"]
    assert "Files found:" in stdout
    assert "Script completed" in stdout
