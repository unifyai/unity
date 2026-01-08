import pytest

from unity.manager_registry import ManagerRegistry
from tests.helpers import _handle_project

# Import a representative subset of managers – covering each family that uses
# SingletonABCMeta under the hood.  The *Simulated* variants are intentionally
# avoided to exercise the *real* classes while keeping test runtime low.
from unity.contact_manager.contact_manager import ContactManager
from unity.knowledge_manager.knowledge_manager import KnowledgeManager
from unity.memory_manager.memory_manager import MemoryManager
from unity.transcript_manager.transcript_manager import TranscriptManager
from unity.task_scheduler.task_scheduler import TaskScheduler


# ---------------------------------------------------------------------------
#  Helper – parameterisation over the concrete manager classes
# ---------------------------------------------------------------------------
MANAGER_CLASSES = [
    ContactManager,
    KnowledgeManager,
    MemoryManager,
    TranscriptManager,
    TaskScheduler,
]


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize("manager_cls", MANAGER_CLASSES)
async def test_is_singleton(manager_cls):
    """Every concrete *Manager* class must behave as a *singleton*."""

    first = manager_cls()
    second = manager_cls()

    # Both instantiations must return the *exact* same object
    assert (
        first is second
    ), f"{manager_cls.__name__} did not return a singleton instance"

    # The central registry must return the same object too
    assert ManagerRegistry.get_instance(manager_cls) is first


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize("manager_cls", MANAGER_CLASSES)
async def test_clear_registry(manager_cls):
    """After `ManagerRegistry.clear` a brand-new instance should be created."""

    original = manager_cls()
    ManagerRegistry.clear()
    replacement = manager_cls()

    assert (
        original is not replacement
    ), f"{manager_cls.__name__} produced the same instance even after clearing the registry"


@pytest.mark.asyncio
@_handle_project
async def test_composition():
    """All references to the same manager BETWEEN different managers must return the *same* instance."""

    memory_manager = MemoryManager()
    contact_manager = memory_manager._contact_manager
    transcript_manager = memory_manager._transcript_manager
    knowledge_manager = memory_manager._knowledge_manager
    task_scheduler = memory_manager._task_scheduler

    assert contact_manager is ContactManager()
    assert transcript_manager is TranscriptManager()
    assert knowledge_manager is KnowledgeManager()
    assert task_scheduler is TaskScheduler()

    ManagerRegistry.clear()

    contact_manager = ContactManager()
    transcript_manager = TranscriptManager()
    knowledge_manager = KnowledgeManager()
    task_scheduler = TaskScheduler()

    memory_manager = MemoryManager()

    assert contact_manager is memory_manager._contact_manager
    assert transcript_manager is memory_manager._transcript_manager
    assert knowledge_manager is memory_manager._knowledge_manager
    assert task_scheduler is memory_manager._task_scheduler
