import unify
from typing import List, Dict
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
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
    def is_setup_complete(cls) -> bool:
        return cls._setup_complete

    @classmethod
    def setup(cls):
        if cls.is_setup_complete():
            return

        available_contexts = unify.get_contexts()
        context_names = list(available_contexts.keys())

        all_contexts = []
        for manager in ContextHandler.get_managers():
            all_contexts.extend(
                ContextHandler.get_contexts_for_manager(manager, context_names),
            )

        if len(all_contexts) <= 0:
            return

        def create_context_wrapper(context: Dict):
            start_time = time.time()
            try:
                filtered_context = {k: v for k, v in context.items() if k != "fields"}
                result = create_context(**filtered_context)
                if "fields" in context:
                    create_fields(context["fields"], context=context["name"])
            except Exception as e:
                print(f"Error creating context {context['name']}: {e}")
                return None
            print(
                f"Time taken for {context['name']}: {time.time() - start_time} seconds",
            )
            return context["name"]

        with ThreadPoolExecutor() as executor:
            futures = []

            for context in all_contexts:
                futures.append(
                    executor.submit(create_context_wrapper, context),
                )

            for future in as_completed(futures):
                context_name = future.result()
                print(f"Context {context_name} created")

        cls._setup_complete = True

    @staticmethod
    def get_contexts_for_manager(manager, available_contexts: List[str]):
        out = []
        assert hasattr(manager, "Config"), "Manager must have a Config class attribute"
        assert hasattr(
            manager.Config,
            "required_contexts",
        ), "Config must have a required_contexts class attribute"
        for context in manager.Config.required_contexts:
            if context.name in available_contexts:
                continue
            data = {
                "name": context.name,
                "description": context.description,
                "fields": context.fields if context.fields else None,
                "unique_keys": context.unique_keys if context.unique_keys else None,
                "auto_counting": (
                    context.auto_counting if context.auto_counting else None
                ),
            }
            data = {k: v for k, v in data.items() if v is not None}
            out.append(data)
        return out
