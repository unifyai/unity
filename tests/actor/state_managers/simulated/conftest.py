"""
Fixtures for simulated state manager tests.

Seeds test data into SimulatedDataManager and SimulatedFileManager
so Actor eval tests have data to work with.
"""

from __future__ import annotations

import pytest

from unity.manager_registry import ManagerRegistry


@pytest.fixture(autouse=True)
def seed_simulated_data_manager(request: pytest.FixtureRequest) -> None:
    """
    Autouse fixture to seed SimulatedDataManager with test tables.

    Creates tables referenced by test_data/test_operations.py:
    - repairs: priority, amount, tenant_id, cost
    - monthly_stats: amount, region
    - arrears: amount, tenant_id, overdue
    - tenants: tenant_id, name
    - payments: tenant_id, amount, date
    """
    # Only seed for test_data tests
    if "test_data" not in str(getattr(request.node, "fspath", "")):
        return

    dm = ManagerRegistry.get_data_manager()

    # Skip if not simulated (real DataManager doesn't have _tables)
    if not hasattr(dm, "_tables"):
        return

    # Create repairs table
    dm.create_table(
        "repairs",
        description="Repair requests with priority and cost",
        fields={
            "id": "int",
            "priority": "str",
            "amount": "float",
            "tenant_id": "int",
            "cost": "float",
        },
    )
    dm.insert_rows(
        "Data/repairs",
        [
            {
                "id": 1,
                "priority": "high",
                "amount": 500.0,
                "tenant_id": 101,
                "cost": 450.0,
            },
            {
                "id": 2,
                "priority": "low",
                "amount": 100.0,
                "tenant_id": 102,
                "cost": 80.0,
            },
            {
                "id": 3,
                "priority": "high",
                "amount": 750.0,
                "tenant_id": 103,
                "cost": 700.0,
            },
            {
                "id": 4,
                "priority": "medium",
                "amount": 300.0,
                "tenant_id": 101,
                "cost": 250.0,
            },
            {
                "id": 5,
                "priority": "high",
                "amount": 1200.0,
                "tenant_id": 104,
                "cost": 1100.0,
            },
        ],
    )

    # Create repairs/2024 table for "What is the average repair cost" question
    dm.create_table(
        "Repairs/2024",
        description="2024 repair records",
        fields={"id": "int", "cost": "float", "description": "str"},
    )
    dm.insert_rows(
        "Data/Repairs/2024",
        [
            {"id": 1, "cost": 500.0, "description": "Plumbing repair"},
            {"id": 2, "cost": 300.0, "description": "Electrical work"},
            {"id": 3, "cost": 800.0, "description": "HVAC maintenance"},
            {"id": 4, "cost": 150.0, "description": "Door fix"},
        ],
    )

    # Create monthly_stats table (simple name to match test questions)
    dm.create_table(
        "monthly_stats",
        description="Monthly statistics by region",
        fields={"id": "int", "amount": "float", "region": "str", "month": "str"},
    )
    dm.insert_rows(
        "Data/monthly_stats",
        [
            {"id": 1, "amount": 1500.0, "region": "North", "month": "Jan"},
            {"id": 2, "amount": 800.0, "region": "South", "month": "Jan"},
            {"id": 3, "amount": 2000.0, "region": "North", "month": "Feb"},
            {"id": 4, "amount": 500.0, "region": "East", "month": "Jan"},
            {"id": 5, "amount": 1200.0, "region": "South", "month": "Feb"},
        ],
    )

    # Also create Pipeline/monthly_stats for "Get all rows from Data/Pipeline/monthly_stats" question
    dm.create_table(
        "Pipeline/monthly_stats",
        description="Monthly statistics by region (Pipeline)",
        fields={"id": "int", "amount": "float", "region": "str", "month": "str"},
    )
    dm.insert_rows(
        "Data/Pipeline/monthly_stats",
        [
            {"id": 1, "amount": 1500.0, "region": "North", "month": "Jan"},
            {"id": 2, "amount": 800.0, "region": "South", "month": "Jan"},
            {"id": 3, "amount": 2000.0, "region": "North", "month": "Feb"},
            {"id": 4, "amount": 500.0, "region": "East", "month": "Jan"},
            {"id": 5, "amount": 1200.0, "region": "South", "month": "Feb"},
        ],
    )

    # Create arrears table
    dm.create_table(
        "arrears",
        description="Tenant arrears and overdue amounts",
        fields={"id": "int", "tenant_id": "int", "amount": "float", "overdue": "bool"},
    )
    dm.insert_rows(
        "Data/arrears",
        [
            {"id": 1, "tenant_id": 101, "amount": 600.0, "overdue": True},
            {"id": 2, "tenant_id": 102, "amount": 200.0, "overdue": False},
            {"id": 3, "tenant_id": 103, "amount": 800.0, "overdue": True},
            {"id": 4, "tenant_id": 104, "amount": 100.0, "overdue": False},
        ],
    )

    # Create tenants table
    dm.create_table(
        "tenants",
        description="Tenant information",
        fields={"tenant_id": "int", "name": "str", "unit": "str"},
    )
    dm.insert_rows(
        "Data/tenants",
        [
            {"tenant_id": 101, "name": "Alice Smith", "unit": "A1"},
            {"tenant_id": 102, "name": "Bob Jones", "unit": "B2"},
            {"tenant_id": 103, "name": "Carol White", "unit": "C3"},
            {"tenant_id": 104, "name": "David Brown", "unit": "D4"},
        ],
    )

    # Create payments table
    dm.create_table(
        "payments",
        description="Payment records",
        fields={"id": "int", "tenant_id": "int", "amount": "float", "date": "str"},
    )
    dm.insert_rows(
        "Data/payments",
        [
            {"id": 1, "tenant_id": 101, "amount": 400.0, "date": "2024-01-15"},
            {"id": 2, "tenant_id": 102, "amount": 200.0, "date": "2024-01-15"},
            {"id": 3, "tenant_id": 103, "amount": 300.0, "date": "2024-02-01"},
            {"id": 4, "tenant_id": 101, "amount": 200.0, "date": "2024-02-15"},
        ],
    )


@pytest.fixture(autouse=True)
def seed_simulated_file_manager(request: pytest.FixtureRequest) -> None:
    """
    Autouse fixture to seed SimulatedFileManager with test files.

    Creates files referenced by test_files/test_ask.py:
    - /reports/ directory with files
    - /data/ directory with CSV files
    - /docs/ directory with documents
    """
    # Only seed for test_files tests
    if "test_files" not in str(getattr(request.node, "fspath", "")):
        return

    fm = ManagerRegistry.get_file_manager()

    # Skip if not simulated (real FileManager doesn't have add_simulated_file)
    if not hasattr(fm, "add_simulated_file"):
        return

    # Use add_simulated_file method which properly sets up file metadata
    # Reports directory
    fm.add_simulated_file(
        filename="/reports/summary.pdf",
        records=1,
        full_text="Quarterly financial summary report. Key findings: Revenue up 15%, costs down 8%. This document covers Q4 2024 performance metrics.",
        description="Q4 2024 financial summary report",
    )
    fm.add_simulated_file(
        filename="/reports/monthly.xlsx",
        records=50,
        full_text="Monthly report spreadsheet with columns: Date, Revenue, Expenses, Net, Region, Category. Contains financial data organized by month.",
        description="Monthly financial data spreadsheet",
    )
    fm.add_simulated_file(
        filename="/reports/Q1_2024.pdf",
        records=1,
        full_text="Q1 2024 quarterly revenue report. Total revenue: $1.2M. Growth: 12% YoY.",
        description="Q1 2024 quarterly revenue report",
    )

    # Data directory
    fm.add_simulated_file(
        filename="/data/Q4_2024.csv",
        records=100,
        full_text="date,revenue,region\n2024-10-01,5000,North\n2024-11-01,6000,South\n2024-12-01,7500,West",
        description="Q4 2024 revenue data by region",
    )
    fm.add_simulated_file(
        filename="/data/sales.csv",
        records=200,
        full_text="product,quantity,price,date\nWidget A,100,25.00,2024-01-15\nWidget B,50,35.00,2024-01-16\nWidget C,75,45.00,2024-01-17",
        description="Sales transaction data",
    )

    # Docs directory
    fm.add_simulated_file(
        filename="/docs/meeting_notes.docx",
        records=1,
        full_text="Meeting notes from Jan 15, 2024.\n\nAttendees: Team leads from Engineering, Product, Sales.\n\nTopics discussed:\n1. Q1 planning and resource allocation\n2. Budget review and cost optimization\n3. Product roadmap updates",
        description="Team meeting notes from January 2024",
    )

    # Documents folder (for search query "quarterly revenue in the documents folder")
    fm.add_simulated_file(
        filename="/documents/quarterly_revenue_2024.pdf",
        records=1,
        full_text="Quarterly revenue analysis for 2024. Q1: $1.2M, Q2: $1.4M, Q3: $1.6M, Q4: $1.8M projected.",
        description="2024 quarterly revenue analysis",
    )
