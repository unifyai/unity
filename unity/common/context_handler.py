import unify
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from unity.common.state_managers import BaseStateManager
from unify import create_context, create_fields
from pydantic import BaseModel
from typing import Optional, Any


class TableContext(BaseModel):
    name: str
    description: str
    fields: Optional[Any] = None
    unique_keys: Optional[Dict[str, str]] = None
    auto_counting: Optional[Dict[str, Optional[str]]] = None


class ContextHandler:
    _setup_complete = False
    _available_contexts = {}

    @classmethod
    def get_context(cls, manager: BaseStateManager, ctx_name: str) -> Optional[str]:
        key = (type(manager).__name__, ctx_name)
        ret = cls._available_contexts.get(key)
        if ret is None:
            active_context = unify.get_active_context()
            assert (
                active_context["read"] == active_context["write"]
            ), "Read and write contexts must be the same"
            available_contexts = unify.get_contexts()
            context_names = list(available_contexts.keys())
            contexts = cls.get_contexts_for_manager(
                manager,
                active_context["read"],
                context_names,
            )
            for context in contexts:
                cls.create_context_wrapper(context)
            ret = cls._available_contexts.get(key)

        return ret

    @classmethod
    def get_managers(cls):
        # TODO: tbh not the best approach to get the managers, but it works for now

        from unity.contact_manager.contact_manager import ContactManager
        from unity.knowledge_manager.knowledge_manager import KnowledgeManager
        from unity.transcript_manager.transcript_manager import TranscriptManager
        from unity.task_scheduler.task_scheduler import TaskScheduler
        from unity.guidance_manager.guidance_manager import GuidanceManager
        from unity.secret_manager.secret_manager import SecretManager
        from unity.web_searcher.web_searcher import WebSearcher

        return [
            ContactManager,
            KnowledgeManager,
            TranscriptManager,
            TaskScheduler,
            GuidanceManager,
            SecretManager,
            WebSearcher,
        ]

    @classmethod
    def create_context_wrapper(cls, entry: Dict):
        try:
            create_context(
                entry["name"],
                description=entry["table_context"].description,
                unique_keys=entry["table_context"].unique_keys,
                auto_counting=entry["table_context"].auto_counting,
            )
            if entry["table_context"].fields:
                create_fields(entry["table_context"].fields, context=entry["name"])
        except Exception as e:
            print(f"Error creating context {entry['name']}: {e}")
            return None

        cls._available_contexts[(entry["manager"], entry["table_context"].name)] = (
            entry["name"]
        )

        return entry["name"]

    @classmethod
    def setup(cls):
        if cls._setup_complete:
            return

        available_contexts = unify.get_contexts()
        context_names = list(available_contexts.keys())

        active_context = unify.get_active_context()
        assert (
            active_context["read"] == active_context["write"]
        ), "Read and write contexts must be the same"

        all_contexts = []
        for manager in ContextHandler.get_managers():
            all_contexts.extend(
                cls.get_contexts_for_manager(
                    manager,
                    active_context["read"],
                    context_names,
                ),
            )

        if len(all_contexts) <= 0:
            return

        with ThreadPoolExecutor() as executor:
            futures = []

            for entry in all_contexts:
                futures.append(
                    executor.submit(cls.create_context_wrapper, entry),
                )

        cls._setup_complete = True

    @classmethod
    def get_contexts_for_manager(
        cls,
        manager,
        current_context: str,
        available_contexts: List[str],
    ):
        assert hasattr(manager, "Config"), "Manager must have a Config class attribute"
        assert hasattr(
            manager.Config,
            "required_contexts",
        ), "Config must have a required_contexts class attribute"

        out = []
        manager_name: str
        try:
            manager_name = manager.__name__
        except:
            manager_name = type(manager).__name__

        for context in manager.Config.required_contexts:
            _context_name = f"{current_context}/{context.name}"
            if _context_name in available_contexts:
                cls._available_contexts[(manager_name, context.name)] = _context_name
                continue
            data = {
                "manager": manager_name,
                "name": _context_name,
                "table_context": context,
            }
            out.append(data)
        return out
