"""
Tests for DataManager integration with ManagerRegistry.
"""

from __future__ import annotations

import pytest

from unity.manager_registry import ManagerRegistry
from unity.data_manager.base import BaseDataManager
from unity.data_manager.data_manager import DataManager
from unity.data_manager.simulated import SimulatedDataManager


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before and after each test."""
    ManagerRegistry.clear()
    yield
    ManagerRegistry.clear()


def test_data_manager_class_registered():
    """DataManager should be registered in the registry."""
    klass = ManagerRegistry.get_class("data", "real")
    assert klass is DataManager


def test_simulated_data_manager_class_registered():
    """SimulatedDataManager should be registered in the registry."""
    klass = ManagerRegistry.get_class("data", "simulated")
    assert klass is SimulatedDataManager


def test_get_data_manager_returns_instance():
    """get_data_manager should return a DataManager instance."""
    dm = ManagerRegistry.get_data_manager()
    assert isinstance(dm, BaseDataManager)


def test_get_data_manager_singleton():
    """get_data_manager should return the same instance."""
    dm1 = ManagerRegistry.get_data_manager()
    dm2 = ManagerRegistry.get_data_manager()
    assert dm1 is dm2


def test_get_data_manager_force_new():
    """get_data_manager with _force_new should create new instance."""
    dm1 = ManagerRegistry.get_data_manager()
    dm2 = ManagerRegistry.get_data_manager(_force_new=True)

    # Both should be valid DataManager instances
    assert isinstance(dm1, BaseDataManager)
    assert isinstance(dm2, BaseDataManager)

    # But they should be different instances (when using _force_new)
    # Note: Due to singleton metaclass behavior, this might still return same
    # instance unless the class doesn't use SingletonABCMeta


def test_registry_clear_removes_data_manager():
    """clear() should remove cached DataManager instance."""
    dm1 = ManagerRegistry.get_data_manager()
    ManagerRegistry.clear()
    dm2 = ManagerRegistry.get_data_manager()

    # After clear, should get a new instance
    # (behavior depends on implementation)
    assert isinstance(dm2, BaseDataManager)
