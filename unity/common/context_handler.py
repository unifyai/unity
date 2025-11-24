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

    @staticmethod
    def _get_active_context() -> str:
        active_context = unify.get_active_context()
        assert (
            active_context["read"] == active_context["write"]
        ), "Read and write contexts must be the same"
        return active_context["read"]

    @staticmethod
    def _get_available_contexts() -> List[str]:
        return list(unify.get_contexts().keys())

    @classmethod
    def get_context(cls, manager: BaseStateManager, ctx_name: str) -> Optional[str]:
        key = (type(manager).__name__, ctx_name)
        ret = cls._available_contexts.get(key)
        if ret is None:
            active_context = cls._get_active_context()
            contexts = cls.get_contexts_for_manager(manager, active_context)
            available_contexts = cls._get_available_contexts()
            for context in contexts:
                cls.create_context_wrapper(context, available_contexts)
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
    def create_context_wrapper(cls, entry: Dict, remote_contexts: List[str]):
        table = entry["table_context"]
        try:
            if entry["name"] not in remote_contexts:
                create_context(
                    entry["name"],
                    description=table.description,
                    unique_keys=table.unique_keys,
                    auto_counting=table.auto_counting,
                )
            # TODO: No need to check current fields, this has no effect if fields are already created
            # possibly can be eliminated if get_fields returns the context for the fields
            if table.fields:
                create_fields(table.fields, context=entry["name"])
        except Exception as e:
            print(f"Error creating context {entry['name']}: {e}")
            return None

        cls._available_contexts[(entry["manager"], table.name)] = entry["name"]

        return entry["name"]

    @classmethod
    def setup(cls):
        if cls._setup_complete:
            return

        current_context = cls._get_active_context()
        available_contexts = cls._get_available_contexts()

        all_contexts = []
        for manager in ContextHandler.get_managers():
            all_contexts.extend(
                cls.get_contexts_for_manager(
                    manager,
                    current_context,
                ),
            )

        with ThreadPoolExecutor() as executor:
            futures = []

            for entry in all_contexts:
                futures.append(
                    executor.submit(
                        cls.create_context_wrapper,
                        entry,
                        available_contexts,
                    ),
                )

        cls._setup_complete = True

    @classmethod
    def get_contexts_for_manager(
        cls,
        manager,
        current_context: str,
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
            data = {
                "manager": manager_name,
                "name": f"{current_context}/{context.name}",
                "table_context": context,
            }
            out.append(data)
        return out
