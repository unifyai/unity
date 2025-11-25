import unify
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    foreign_keys: Optional[List[Dict[str, Any]]] = None


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

    @staticmethod
    def _get_manager_name(manager: BaseStateManager) -> str:
        try:
            return manager.__name__
        except:
            return type(manager).__name__

    @classmethod
    def get_context(cls, manager: BaseStateManager, ctx_name: str) -> Optional[str]:
        manager_name = cls._get_manager_name(manager)
        key = (manager_name, ctx_name)
        ret = cls._available_contexts.get(key)
        if ret is None:
            active_context = cls._get_active_context()
            contexts = cls.get_contexts_for_manager(manager, active_context)
            available_contexts = cls._get_available_contexts()
            ret = cls.create_context_wrapper(
                manager_name,
                contexts[ctx_name],
                available_contexts,
            )

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
        from unity.image_manager.image_manager import ImageManager
        from unity.file_manager.managers.file_manager import FileManager
        from unity.function_manager.function_manager import FunctionManager

        return [
            ContactManager,
            KnowledgeManager,
            TranscriptManager,
            TaskScheduler,
            ImageManager,
            GuidanceManager,
            SecretManager,
            WebSearcher,
            # FileManager,
            FunctionManager,
        ]

    @classmethod
    def create_context_wrapper(
        cls,
        manager_name: str,
        entry: Dict,
        remote_contexts: List[str],
    ):
        table = entry["table_context"]
        target_name = entry["resolved_name"]
        try:
            if target_name not in remote_contexts:
                create_context(
                    target_name,
                    description=table.description,
                    unique_keys=table.unique_keys,
                    auto_counting=table.auto_counting,
                    foreign_keys=table.foreign_keys,
                )
            # TODO: No need to check current fields, this has no effect if fields are already created
            # possibly can be eliminated if get_fields returns the context for the fields
            if table.fields:
                create_fields(table.fields, context=target_name)
        except Exception as e:
            return None

        cls._available_contexts[(manager_name, table.name)] = target_name

        return target_name

    @classmethod
    def setup(cls):
        if cls._setup_complete:
            return

        current_context = cls._get_active_context()
        available_contexts = cls._get_available_contexts()

        with ThreadPoolExecutor() as executor:
            futures = []
            for manager in cls.get_managers():
                manager_name = cls._get_manager_name(manager)
                for _, entry in cls.get_contexts_for_manager(
                    manager,
                    current_context,
                ).items():
                    futures.append(
                        executor.submit(
                            cls.create_context_wrapper,
                            manager_name,
                            entry,
                            available_contexts,
                        ),
                    )

            for future in as_completed(futures):
                future.result()

        cls._setup_complete = True

    @classmethod
    def get_contexts_for_manager(
        cls,
        manager,
        current_context: str,
    ) -> Dict[str, Dict]:
        assert hasattr(manager, "Config"), "Manager must have a Config class attribute"
        assert hasattr(
            manager.Config,
            "required_contexts",
        ), "Config must have a required_contexts class attribute"

        out = {}

        for context in manager.Config.required_contexts:
            if context.foreign_keys:
                for foreign_key in context.foreign_keys:
                    foreign_key["references"] = (
                        f"{current_context}/{foreign_key['references']}"
                    )
            data = {
                "resolved_name": f"{current_context}/{context.name}",
                "table_context": context,
            }
            out[context.name] = data
        return out
