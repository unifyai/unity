"""
Fixtures for DataManager tests.

Most tests use SimulatedDataManager which requires no backend.
Integration tests that need the real backend can use the real_data_manager fixture.
"""

from __future__ import annotations

import pytest

from unity.data_manager.simulated import SimulatedDataManager
from unity.manager_registry import ManagerRegistry


@pytest.fixture
def simulated_dm() -> SimulatedDataManager:
    """Provide a fresh SimulatedDataManager instance for each test."""
    dm = SimulatedDataManager()
    yield dm
    dm.clear()


@pytest.fixture
def seeded_dm() -> SimulatedDataManager:
    """Provide a SimulatedDataManager pre-seeded with test data."""
    dm = SimulatedDataManager()

    # Create test tables
    dm.create_table(
        "test/products",
        description="Product catalog",
        fields={"id": "int", "name": "str", "price": "float", "category": "str"},
    )
    dm.create_table(
        "test/orders",
        description="Order records",
        fields={
            "order_id": "int",
            "product_id": "int",
            "quantity": "int",
            "status": "str",
        },
    )
    dm.create_table(
        "test/customers",
        description="Customer records",
        fields={"customer_id": "int", "name": "str", "region": "str"},
    )

    # Seed products
    dm.insert_rows(
        "test/products",
        [
            {"id": 1, "name": "Widget A", "price": 10.0, "category": "widgets"},
            {"id": 2, "name": "Widget B", "price": 20.0, "category": "widgets"},
            {"id": 3, "name": "Gadget X", "price": 50.0, "category": "gadgets"},
            {"id": 4, "name": "Gadget Y", "price": 75.0, "category": "gadgets"},
            {"id": 5, "name": "Tool Z", "price": 100.0, "category": "tools"},
        ],
    )

    # Seed orders
    dm.insert_rows(
        "test/orders",
        [
            {"order_id": 101, "product_id": 1, "quantity": 5, "status": "shipped"},
            {"order_id": 102, "product_id": 2, "quantity": 3, "status": "pending"},
            {"order_id": 103, "product_id": 3, "quantity": 2, "status": "shipped"},
            {"order_id": 104, "product_id": 1, "quantity": 10, "status": "delivered"},
            {"order_id": 105, "product_id": 4, "quantity": 1, "status": "pending"},
        ],
    )

    # Seed customers
    dm.insert_rows(
        "test/customers",
        [
            {"customer_id": 1, "name": "Alice", "region": "East"},
            {"customer_id": 2, "name": "Bob", "region": "West"},
            {"customer_id": 3, "name": "Carol", "region": "East"},
        ],
    )

    yield dm
    dm.clear()


@pytest.fixture
def real_data_manager():
    """Provide the real DataManager for integration tests.

    Note: This requires a working backend connection.
    """
    ManagerRegistry.clear()
    dm = ManagerRegistry.get_data_manager()
    yield dm
    ManagerRegistry.clear()
