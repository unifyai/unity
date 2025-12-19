import pytest

from unity.manager_registry import ManagerRegistry
from tests.helpers import _handle_project

# Import simulated manager variants – each relies on `SingletonABCMeta` via their base class.
from unity.contact_manager.simulated import SimulatedContactManager
from unity.knowledge_manager.simulated import SimulatedKnowledgeManager
from unity.memory_manager.simulated import SimulatedMemoryManager
from unity.transcript_manager.simulated import SimulatedTranscriptManager
from unity.task_scheduler.simulated import SimulatedTaskScheduler

# ---------------------------------------------------------------------------
#  Helper – parameterisation over the concrete simulated manager classes
# ---------------------------------------------------------------------------
MANAGER_CLASSES = [
    SimulatedContactManager,
    SimulatedKnowledgeManager,
    SimulatedMemoryManager,
    SimulatedTranscriptManager,
    SimulatedTaskScheduler,
]


@pytest.mark.asyncio
@_handle_project
@pytest.mark.parametrize("manager_cls", MANAGER_CLASSES)
async def test_is_singleton(manager_cls):
    """Every *simulated* manager class must behave as a *singleton*."""

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
    """After `ManagerRegistry.clear` a brand-new instance should be created (simulated)."""

    original = manager_cls()
    ManagerRegistry.clear()
    replacement = manager_cls()

    assert (
        original is not replacement
    ), f"{manager_cls.__name__} produced the same instance even after clearing the registry"


@pytest.mark.asyncio
@_handle_project
async def test_composition():
    """All references to the same simulated manager BETWEEN managers must return the *same* instance."""

    memory_manager = SimulatedMemoryManager()
    contact_manager = memory_manager._contact_manager
    transcript_manager = memory_manager._transcript_manager
    knowledge_manager = memory_manager._knowledge_manager
    task_scheduler = memory_manager._task_scheduler

    assert contact_manager is SimulatedContactManager()
    assert transcript_manager is SimulatedTranscriptManager()
    assert knowledge_manager is SimulatedKnowledgeManager()
    assert task_scheduler is SimulatedTaskScheduler()

    ManagerRegistry.clear()

    contact_manager = SimulatedContactManager()
    transcript_manager = SimulatedTranscriptManager()
    knowledge_manager = SimulatedKnowledgeManager()
    task_scheduler = SimulatedTaskScheduler()

    memory_manager = SimulatedMemoryManager()

    assert contact_manager is memory_manager._contact_manager
    assert transcript_manager is memory_manager._transcript_manager
    assert knowledge_manager is memory_manager._knowledge_manager
    assert task_scheduler is memory_manager._task_scheduler
