from __future__ import annotations

import ast
import inspect
from typing import Any, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


class SimulatedLineage:
    """Helpers for building nested labels and previews for simulated flows."""

    PREVIEW_LIMIT = 120

    @staticmethod
    def parent_lineage() -> List[str]:
        try:
            # Local import to avoid import cycles at module import time
            from unity.common._async_tool.loop_config import (
                TOOL_LOOP_LINEAGE,
            )  # noqa: WPS433
        except Exception:
            return []
        try:
            val = TOOL_LOOP_LINEAGE.get([])
            return list(val) if isinstance(val, list) else []
        except Exception:
            return []

    @staticmethod
    def has_outer() -> bool:
        try:
            return bool(SimulatedLineage.parent_lineage())
        except Exception:
            return False

    @staticmethod
    def make_label(segment: str) -> str:
        """Compose a nested label like '<outer...>->Segment(abcd)'."""
        from secrets import token_hex  # noqa: WPS433

        try:
            parts = SimulatedLineage.parent_lineage()
            base = "->".join([*parts, segment]) if parts else segment
        except Exception:
            base = segment
        return f"{base}({token_hex(2)})"

    @staticmethod
    def question_label(parent_label: str) -> str:
        """Build a concise child label 'Question(<parent>)(abcd)'."""
        from secrets import token_hex  # noqa: WPS433

        return f"Question({parent_label})({token_hex(2)})"

    # --- New helpers for consistent session suffix reuse ---------------------
    @staticmethod
    def extract_suffix(label: str) -> "str | None":
        """
        Return the trailing '(xxxx)' hex suffix from a label when present.
        """
        s = str(label or "").strip()
        if not s.endswith(")"):
            return None
        try:
            open_idx = s.rfind("(")
            if open_idx == -1:
                return None
            return s[open_idx + 1 : -1] or None
        except Exception:
            return None

    @staticmethod
    def make_label_with_suffix(segment: str, suffix: str) -> str:
        """
        Compose '<outer...>->Segment(suffix)' using the provided suffix.
        """
        try:
            parts = SimulatedLineage.parent_lineage()
            base = "->".join([*parts, segment]) if parts else segment
        except Exception:
            base = segment
        suf = str(suffix or "").strip()
        return f"{base}({suf})" if suf else SimulatedLineage.make_label(segment)

    @staticmethod
    def preview(text: str, limit: int = PREVIEW_LIMIT) -> str:
        s = str(text or "")
        return s if len(s) <= int(limit) else f"{s[:int(limit)]}…"


class SimulatedLog:
    """Small wrapper for consistent iconised request/steering logs."""

    _ICONS = {
        "ask": "❓",
        "update": "📝",
        "execute": "🎬",
        "act": "🎬",
        "interject": "💬",
        "pause": "⏸️",
        "resume": "▶️",
        "stop": "🛑",
        # simulated-only convenience
        "clar_req": "❓",
        "clar_ans": "💬",
        "notification": "🔔",
        # session lifecycle
        "session_start": "🚀",
        "session_end": "🏁",
    }
    _VERBS = {
        "ask": "Ask requested",
        "update": "Update requested",
        "execute": "Execute requested",
        "act": "Act requested",
        "interject": "Interject requested",
        "pause": "Pause requested",
        "resume": "Resume requested",
        "stop": "Stop requested",
        # simulated-only convenience
        "clar_req": "Clarification requested",
        "clar_ans": "Clarification answer received",
        "notification": "Notification",
        # session lifecycle
        "session_start": "Session started",
        "session_end": "Session ended",
    }

    @staticmethod
    def log_request(kind: str, label: str, text: str = "") -> None:
        try:
            from unity.constants import LOGGER  # noqa: WPS433
        except Exception:
            return
        try:
            icon = SimulatedLog._ICONS.get(kind, "ℹ️")
            verb = SimulatedLog._VERBS.get(kind, "Requested")
            suffix = ""
            if kind in {"ask", "update", "act", "interject"}:
                prev = SimulatedLineage.preview(text)
                if prev:
                    suffix = f": {prev}"
            LOGGER.info(f"{icon} [{label}] {verb}{suffix}")
        except Exception:
            # Never let logging break control flow
            pass

    @staticmethod
    def log_clarification_request(label: str, question: str) -> None:
        """Emit a standardised clarification-request log line."""
        try:
            from unity.constants import LOGGER  # noqa: WPS433
        except Exception:
            return
        try:
            q = SimulatedLineage.preview(question)
            LOGGER.info(f"❓ [{label}] Clarification requested: {q}")
        except Exception:
            pass

    @staticmethod
    def log_clarification_answer(label: str, answer: str) -> None:
        """Emit a standardised clarification-answer log line."""
        try:
            from unity.constants import LOGGER  # noqa: WPS433
        except Exception:
            return
        try:
            a = SimulatedLineage.preview(answer)
            LOGGER.info(f"💬 [{label}] Clarification answer received: {a}")
        except Exception:
            pass

    @staticmethod
    def log_notification(label: str, message: str) -> None:
        """Emit a standardised notification log line."""
        try:
            from unity.constants import LOGGER  # noqa: WPS433
        except Exception:
            return
        try:
            m = SimulatedLineage.preview(message)
            LOGGER.info(f"🔔 [{label}] Notification: {m}")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Tool-call style logging helpers (gated by parent lineage)
# ─────────────────────────────────────────────────────────────────────────────
def maybe_tool_log_scheduled(segment: str, method: str, args: dict):
    """
    Emit a standardized 'ToolCall Scheduled' log line when there is no parent lineage.
    Returns (label, call_id, t0) on success; otherwise None.
    """
    try:
        if SimulatedLineage.has_outer():
            return None
        from unity.constants import LOGGER  # noqa: WPS433
        import json as _json  # noqa: WPS433
        import time as _time  # noqa: WPS433

        label = SimulatedLineage.make_label(segment)
        cid = SimulatedLineage.extract_suffix(label) or ""
        try:
            LOGGER.info(
                f"🛠️ [{label}] ToolCall Scheduled | args={_json.dumps(args)}",
            )
        except Exception:
            pass
        return label, cid, _time.perf_counter()
    except Exception:
        return None


def maybe_tool_log_scheduled_with_label(label: str, method: str, args: dict):
    """
    Emit 'ToolCall Scheduled' using a precomputed label when there is no parent lineage.
    Returns (label, call_id, t0) on success; otherwise None.
    """
    try:
        if SimulatedLineage.has_outer():
            return None
        from unity.constants import LOGGER  # noqa: WPS433
        import json as _json  # noqa: WPS433
        import time as _time  # noqa: WPS433

        cid = SimulatedLineage.extract_suffix(label) or ""
        try:
            LOGGER.info(
                f"🛠️ [{label}] ToolCall Scheduled | args={_json.dumps(args)}",
            )
        except Exception:
            pass
        return label, cid, _time.perf_counter()
    except Exception:
        return None


def maybe_tool_log_completed(
    label: str,
    cid: str,
    method: str,
    result: dict,
    t0: float,
) -> None:
    """
    Emit a standardized 'ToolCall Completed' log line when there is no parent lineage.
    """
    try:
        if SimulatedLineage.has_outer():
            return
        from unity.constants import LOGGER  # noqa: WPS433
        import json as _json  # noqa: WPS433
        import time as _time  # noqa: WPS433

        dt = _time.perf_counter() - float(t0)
        try:
            LOGGER.info(
                f"✅ [{label}] ToolCall Completed in {dt:.2f}s | result={_json.dumps(result)}",
            )
        except Exception:
            pass
    except Exception:
        pass


async def simulated_llm_roundtrip(
    llm: Any,
    *,
    label: str,
    prompt: str,
    response_format: Any = None,
) -> str:
    """Unified 'LLM simulating' roundtrip with console logging.

    LLM I/O debugging is now handled by hooks installed on the unify client.

    Parameters
    ----------
    llm : Any
        The LLM client to use for generation.
    label : str
        Human-readable label for logging.
    prompt : str
        The prompt to send to the LLM.
    response_format : Type[BaseModel] | None
        Optional Pydantic model for structured output. When provided,
        the LLM's response_format is set before generation and reset after.
    """
    try:
        from unity.constants import LOGGER  # noqa: WPS433
    except Exception:
        LOGGER = None  # type: ignore

    import time as _time  # noqa: WPS433

    try:
        if LOGGER is not None:
            LOGGER.info(f"🔄 [{label}] LLM simulating…")
    except Exception:
        pass
    t0 = _time.perf_counter()

    # Set response_format if provided
    if response_format is not None:
        try:
            llm.set_response_format(response_format)
        except Exception:
            pass
    try:
        answer = await llm.generate(prompt)
    finally:
        if response_format is not None:
            try:
                llm.reset_response_format()
            except Exception:
                pass
    dt_ms = int((_time.perf_counter() - t0) * 1000)

    try:
        if LOGGER is not None:
            if SimulatedLineage.has_outer():
                LOGGER.info(f"✅ [{label}] LLM replied in {dt_ms} ms")
            else:
                _ans_preview = str(answer)
                if len(_ans_preview) > 800:
                    _ans_preview = _ans_preview[:800] + "…"
                LOGGER.info(f"✅ [{label}] LLM replied in {dt_ms} ms:\n{_ans_preview}")
    except Exception:
        pass

    return answer  # type: ignore[return-value]


class SimulatedHandleMixin:
    """Lightweight mixin to standardise steering logs for simulated handles."""

    # Derived classes are expected to set: self._log_label : str

    def _log_interject(self, message: str) -> None:
        try:
            SimulatedLog.log_request(
                "interject",
                getattr(self, "_log_label", "handle"),
                str(message),
            )
        except Exception:
            pass

    def _log_pause(self) -> None:
        try:
            SimulatedLog.log_request("pause", getattr(self, "_log_label", "handle"))
        except Exception:
            pass

    def _log_resume(self) -> None:
        try:
            SimulatedLog.log_request("resume", getattr(self, "_log_label", "handle"))
        except Exception:
            pass

    def _log_stop(self, reason: str | None) -> None:
        try:
            from unity.constants import LOGGER  # noqa: WPS433
        except Exception:
            return
        try:
            suffix = f" – reason: {reason}" if reason else ""
            LOGGER.info(
                f"🛑 [{getattr(self, '_log_label', 'handle')}] Stop requested{suffix}",
            )
        except Exception:
            pass


def _extract_owner_method_pairs(
    cls: Any,
    target_attr: str,
    self_external_map: Dict[str, Any] | None = None,
    extra_class_names: Dict[str, Any] | None = None,
) -> List[Tuple[Any, str]]:
    """Extract (owner_class, method_name) pairs from cls.__init__ for target_attr.

    Finds assignments to self.<target_attr> that are built via methods_to_tool_dict(...)
    and returns each referenced method as (owner_class, method_name).
    """
    try:
        src = inspect.getsource(cls.__init__)
    except Exception:
        return []

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    results: List[Tuple[Any, str]] = []

    def _process_value(value: ast.AST) -> None:
        for node in ast.walk(value):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                (isinstance(func, ast.Name) and func.id == "methods_to_tool_dict")
                or (
                    isinstance(func, ast.Attribute)
                    and func.attr == "methods_to_tool_dict"
                )
            ):
                continue
            for arg in node.args:
                if not isinstance(arg, ast.Attribute):
                    continue
                name = arg.attr
                owner: Any | None = None
                root = arg.value
                # self.<attr> → current class
                if isinstance(root, ast.Name) and root.id == "self":
                    owner = cls
                # ContactManager.<method> or other explicitly named class
                elif (
                    isinstance(root, ast.Name)
                    and extra_class_names
                    and root.id in extra_class_names
                ):
                    owner = extra_class_names[root.id]
                # self._contact_manager.<method> (or other mapped external refs)
                elif (
                    isinstance(root, ast.Attribute)
                    and isinstance(root.value, ast.Name)
                    and root.value.id == "self"
                    and self_external_map
                    and root.attr in self_external_map
                ):
                    owner = self_external_map[root.attr]
                if owner and name:
                    results.append((owner, name))

    class _Visitor(ast.NodeVisitor):
        def _matches_target(self, target: ast.AST) -> bool:
            return (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == target_attr
            )

        def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
            if not node.targets:
                return
            if self._matches_target(node.targets[0]):
                _process_value(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # type: ignore[override]
            if node.value is None:
                return
            if self._matches_target(node.target):
                _process_value(node.value)

    _Visitor().visit(tree)
    return results


def _build_tool_dict(
    pairs: List[Tuple[Any, str]],
    include_class_name_for: set[Any] | None = None,
) -> Dict[str, Any]:
    from unity.common.llm_helpers import methods_to_tool_dict

    include_class_name_for = include_class_name_for or set()
    by_owner: Dict[Any, List[Any]] = {}
    for owner_cls, method_name in pairs:
        try:
            by_owner.setdefault(owner_cls, []).append(getattr(owner_cls, method_name))
        except Exception:
            # Skip if attribute lookup fails; fallback logic will handle empties
            pass

    tools: Dict[str, Any] = {}
    for owner_cls, methods in by_owner.items():
        if not methods:
            continue

        # When class-qualified naming is requested, bind methods to an instance
        # so methods_to_tool_dict can derive `ClassName_method` keys.
        if owner_cls in include_class_name_for:
            try:
                instance = owner_cls()
                bound_methods = [getattr(instance, m.__name__) for m in methods]
            except Exception:
                # Fall back to unbound methods if instantiation fails
                bound_methods = methods
            tools.update(
                methods_to_tool_dict(
                    *bound_methods,
                    include_class_name=True,
                ),
            )
        else:
            tools.update(
                methods_to_tool_dict(
                    *methods,
                    include_class_name=False,
                ),
            )
    return tools


# ─────────────────────────────────────────────────────────────────────────────
# ContactManager mirroring
# ─────────────────────────────────────────────────────────────────────────────


def mirror_contact_manager_tools(kind: str) -> Dict[str, Any]:
    """Build a tool-dict mirroring the real ContactManager's tool lists.

    kind: "ask" or "update". Uses AST reflection with a static fallback.
    """
    from unity.contact_manager.contact_manager import ContactManager
    from unity.common.llm_helpers import methods_to_tool_dict

    target_attr = "_ask_tools" if kind == "ask" else "_update_tools"

    try:
        pairs = _extract_owner_method_pairs(
            ContactManager,
            target_attr,
            self_external_map=None,
            extra_class_names={"ContactManager": ContactManager},
        )
        if pairs:
            # All ContactManager-owned methods; never include class name
            tools = _build_tool_dict(pairs)
            if tools:
                return tools
    except Exception:
        pass

    # Fallback – current canonical tool sets
    if kind == "ask":
        return methods_to_tool_dict(
            ContactManager._list_columns,
            ContactManager.filter_contacts,
            ContactManager._search_contacts,
            ContactManager._reduce,
            include_class_name=False,
        )
    else:
        return methods_to_tool_dict(
            ContactManager.ask,
            ContactManager._create_contact,
            ContactManager.update_contact,
            ContactManager._delete_contact,
            ContactManager._create_custom_column,
            ContactManager._delete_custom_column,
            ContactManager._merge_contacts,
            include_class_name=False,
        )


#
# ─────────────────────────────────────────────────────────────────────────────
# SecretManager mirroring
# ─────────────────────────────────────────────────────────────────────────────
#


def mirror_secret_manager_tools(kind: str) -> Dict[str, Any]:
    """Build a tool-dict mirroring the real SecretManager's tool lists.

    kind: "ask" or "update". Uses AST reflection with a static fallback.
    """
    from unity.secret_manager.secret_manager import SecretManager
    from unity.common.llm_helpers import methods_to_tool_dict

    target_attr = "_ask_tools" if kind == "ask" else "_update_tools"

    try:
        pairs = _extract_owner_method_pairs(
            SecretManager,
            target_attr,
            self_external_map=None,
            extra_class_names={"SecretManager": SecretManager},
        )
        if pairs:
            # All SecretManager-owned methods; never include class name
            tools = _build_tool_dict(pairs)
            if tools:
                return tools
    except Exception:
        pass

    # Fallback – keep in sync with SecretManager.__init__
    if kind == "ask":
        return methods_to_tool_dict(
            SecretManager._list_columns,
            SecretManager._filter_secrets,
            SecretManager._search_secrets,
            SecretManager._list_secret_keys,
            include_class_name=False,
        )
    else:
        return methods_to_tool_dict(
            SecretManager.ask,
            SecretManager._create_secret,
            SecretManager._update_secret,
            SecretManager._delete_secret,
            include_class_name=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TranscriptManager mirroring
# ─────────────────────────────────────────────────────────────────────────────


def mirror_transcript_manager_tools() -> Dict[str, Any]:
    """Build a tool-dict mirroring the real TranscriptManager's tools.

    Uses AST reflection of TranscriptManager.__init__ with a static fallback.
    """
    from unity.common.llm_helpers import methods_to_tool_dict
    from unity.transcript_manager.transcript_manager import TranscriptManager
    from unity.contact_manager.contact_manager import ContactManager

    try:
        pairs = _extract_owner_method_pairs(
            TranscriptManager,
            "_tools",
            self_external_map={"_contact_manager": ContactManager},
        )
        if pairs:
            tools = _build_tool_dict(pairs)
            if tools:
                return tools
    except Exception:
        pass

    # Fallback – current canonical tool set
    return methods_to_tool_dict(
        ContactManager.filter_contacts,
        TranscriptManager._filter_messages,
        TranscriptManager._search_messages,
        TranscriptManager._reduce,
        include_class_name=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TaskScheduler mirroring
# ─────────────────────────────────────────────────────────────────────────────


def mirror_task_scheduler_tools(kind: str) -> Dict[str, Any]:
    """Build a tool-dict mirroring the real TaskScheduler's tool exposure.

    Uses AST reflection of TaskScheduler.__init__ with a static fallback. Ensures
    that external tools like ContactManager.ask retain their class-qualified naming
    by applying include_class_name=True for those only.
    """
    from unity.common.llm_helpers import methods_to_tool_dict
    from unity.task_scheduler.task_scheduler import TaskScheduler
    from unity.contact_manager.contact_manager import ContactManager

    target_attr = "_ask_tools" if kind == "ask" else "_update_tools"

    try:
        pairs = _extract_owner_method_pairs(
            TaskScheduler,
            target_attr,
            self_external_map={"_contact_manager": ContactManager},
        )
        if pairs:
            tools = _build_tool_dict(pairs, include_class_name_for={ContactManager})
            if tools:
                return tools
    except Exception:
        pass

    # Fallback – current canonical tool sets (kept in sync with TaskScheduler)
    if kind == "ask":
        tools: Dict[str, Any] = {}
        tools.update(
            methods_to_tool_dict(
                TaskScheduler._filter_tasks,
                TaskScheduler._search_tasks,
                TaskScheduler._get_queue,
                TaskScheduler._get_queue_for_task,
                TaskScheduler._reduce,
                include_class_name=False,
            ),
        )
        tools.update(
            methods_to_tool_dict(
                ContactManager().ask,
                include_class_name=True,
            ),
        )
        return tools
    else:
        tools = methods_to_tool_dict(
            # Ask entry point is exposed on update side
            TaskScheduler.ask,
            # Read-only task discovery (update flows rely on these too)
            TaskScheduler._filter_tasks,
            TaskScheduler._search_tasks,
            # Creation / deletion / cancellation
            TaskScheduler._create_tasks,
            TaskScheduler._create_task,
            TaskScheduler._delete_task,
            TaskScheduler._cancel_tasks,
            # Queue inspection/manipulation
            TaskScheduler._list_queues,
            TaskScheduler._get_queue,
            TaskScheduler._get_queue_for_task,
            TaskScheduler._reorder_queue,
            TaskScheduler._move_tasks_to_queue,
            TaskScheduler._partition_queue,
            # Reintegration and atomic materialization
            TaskScheduler._reinstate_task_to_previous_queue,
            TaskScheduler._set_queue,
            TaskScheduler._set_schedules_atomic,
            # Unified attribute updater
            TaskScheduler._update_task,
            include_class_name=False,
        )

        tools.update(
            methods_to_tool_dict(
                ContactManager().ask,
                include_class_name=True,
            ),
        )

        return tools


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeManager mirroring
# ─────────────────────────────────────────────────────────────────────────────


def mirror_knowledge_manager_tools(kind: str) -> Dict[str, Any]:
    """Build a tool-dict mirroring the real KnowledgeManager's tools.

    kind: one of "ask", "update", or "refactor". Uses AST reflection of
    KnowledgeManager.__init__ with a static fallback. For "update" we also
    merge tools referenced by "_refactor_tools" because the real implementation
    unions those into "_update_tools".
    """
    from unity.common.llm_helpers import methods_to_tool_dict
    from unity.knowledge_manager.knowledge_manager import KnowledgeManager as KM

    target_map = {
        "ask": "_ask_tools",
        "update": "_update_tools",
        "refactor": "_refactor_tools",
    }
    target_attr = target_map.get(kind, "_ask_tools")

    try:
        pairs = _extract_owner_method_pairs(
            KM,
            target_attr,
            self_external_map=None,
        )
        # Ensure update includes the refactor set as well (it's merged in __init__)
        if kind == "update":
            ref_pairs = _extract_owner_method_pairs(KM, "_refactor_tools")
            pairs.extend(ref_pairs)

        if pairs:
            tools = _build_tool_dict(pairs)
            if tools:
                return tools
    except Exception:
        pass

    # Static fallbacks – keep in sync with KnowledgeManager.__init__
    if kind == "ask":
        return methods_to_tool_dict(
            KM._tables_overview,
            KM._filter,
            KM._search,
            KM._reduce,
            KM._filter_join,
            KM._filter_multi_join,
            include_class_name=False,
        )
    if kind == "refactor":
        return methods_to_tool_dict(
            KM.ask,
            # Tables
            KM._create_table,
            KM._rename_table,
            KM._delete_tables,
            # Columns
            KM._rename_column,
            KM._copy_column,
            KM._move_column,
            KM._delete_column,
            KM._create_empty_column,
            KM._create_derived_column,
            KM._transform_column,
            KM._vectorize_column,
            # Rows
            KM._delete_rows,
            KM._update_rows,
            include_class_name=False,
        )
    # update fallback = refactor tools + _add_rows
    ref_tools = mirror_knowledge_manager_tools("refactor")
    add_rows_tools = methods_to_tool_dict(KM._add_rows, include_class_name=False)
    merged: Dict[str, Any] = {}
    merged.update(ref_tools)
    merged.update(add_rows_tools)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Shared prompt helpers
# ─────────────────────────────────────────────────────────────────────────────


def build_followup_prompt(
    *,
    question: str,
    initial_instruction: str,
    extra_messages: list[str] | None = None,
) -> str:
    """
    Build a standardized follow-up question prompt for simulated handles.

    Ensures consistent wording and formatting across managers and fixes prior typos.
    """
    preamble = (
        f"Your only task is to simulate an answer to the following question: {question}\n\n"
        "There is also an ongoing simulated process with the instructions given below. "
        "Please make your answer realistic and consistent with the context of the simulated task."
    )
    parts: list[str] = [preamble, initial_instruction]
    if extra_messages:
        parts.extend(extra_messages)
    parts.append(f"Question to answer (as a reminder!): {question}")
    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# FileManager mirroring
# ─────────────────────────────────────────────────────────────────────────────


def mirror_file_manager_tools(kind: str) -> Dict[str, Any]:
    """Build a tool-dict mirroring the real FileManager's tool exposure.

    kind: "ask", "ask_about_file", or "organize". Uses AST reflection with a static fallback.
    """
    from unity.common.llm_helpers import methods_to_tool_dict
    from unity.file_manager.managers.local import LocalFileManager as FileManager

    if kind == "ask_about_file":
        target_attr = "_ask_about_file_tools"
    elif kind == "organize":
        target_attr = "_organize_tools"
    else:
        target_attr = "_ask_tools"

    try:
        pairs = _extract_owner_method_pairs(
            FileManager,
            target_attr,
            self_external_map=None,
            extra_class_names={"FileManager": FileManager},
        )
        if pairs:
            tools = _build_tool_dict(pairs)
            if tools:
                return tools
    except Exception:
        pass

    # Fallback – EXACTLY mirror FileManager.__init__ tool exposure
    if kind == "ask":
        return methods_to_tool_dict(
            # Schema discovery
            FileManager._list_columns,
            FileManager._tables_overview,
            FileManager._schema_explain,
            FileManager._file_info,
            # Retrieval helpers
            FileManager._filter_files,
            FileManager._search_files,
            FileManager._reduce,
            # Join/multi-join tools (exposed via ask.multi_table and merged at call-time)
            FileManager._filter_join,
            FileManager._search_join,
            FileManager._filter_multi_join,
            FileManager._search_multi_join,
            # Inventory listing
            FileManager.list,
            # Delegate to file-scoped Q&A when needed
            FileManager.ask_about_file,
            include_class_name=False,
        )
    elif kind == "ask_about_file":
        return methods_to_tool_dict(
            # Schema discovery
            FileManager._file_info,
            FileManager._list_columns,
            FileManager._tables_overview,
            FileManager._schema_explain,
            # Retrieval helpers
            FileManager._filter_files,
            FileManager._search_files,
            FileManager._reduce,
            # Join/multi-join tools for file-scoped analysis
            FileManager._filter_join,
            FileManager._search_join,
            FileManager._filter_multi_join,
            FileManager._search_multi_join,
            include_class_name=False,
        )
    elif kind == "organize":
        return methods_to_tool_dict(
            # Discovery via ask
            FileManager.ask,
            # Mutation tools
            FileManager._rename_file,
            FileManager._move_file,
            FileManager._delete_file,
            # Sync tool (exposed under organize)
            FileManager.sync,
            include_class_name=False,
        )
    else:
        # Default to ask tools
        return methods_to_tool_dict(
            FileManager._list_columns,
            FileManager._tables_overview,
            FileManager._schema_explain,
            FileManager._file_info,
            FileManager._filter_files,
            FileManager._search_files,
            FileManager._reduce,
            FileManager._filter_join,
            FileManager._search_join,
            FileManager._filter_multi_join,
            FileManager._search_multi_join,
            FileManager.list,
            FileManager.ask_about_file,
            include_class_name=False,
        )


def mirror_global_file_manager_tools(kind: str) -> Dict[str, Any]:
    """Build a tool-dict mirroring the real GlobalFileManager's tool exposure.

    kind: "ask" or "organize". Uses AST reflection with a static fallback.
    """
    from unity.common.llm_helpers import methods_to_tool_dict
    from unity.file_manager.global_file_manager import GlobalFileManager

    target_attr = "_ask_tools" if kind == "ask" else "_organize_tools"

    try:
        pairs = _extract_owner_method_pairs(
            GlobalFileManager,
            target_attr,
            self_external_map=None,
            extra_class_names={"GlobalFileManager": GlobalFileManager},
        )
        if pairs:
            tools = _build_tool_dict(pairs)
            if tools:
                return tools
    except Exception as e:
        print(f"mirror_global_file_manager_tools({kind}) failed: {e}")

    # Fallback – align with new delegation-only model
    if kind == "ask":
        # Require listing filesystems first; no low-level ops exposed here
        return methods_to_tool_dict(
            GlobalFileManager._list_filesystems,
            include_class_name=False,
        )
    elif kind == "organize":
        # Organize should have discovery via ask available
        return methods_to_tool_dict(
            GlobalFileManager.ask,
            GlobalFileManager._list_filesystems,
            include_class_name=False,
        )
    else:
        # Default to ask tools
        return methods_to_tool_dict(
            GlobalFileManager._list_filesystems,
            GlobalFileManager._list_columns,
            GlobalFileManager._filter_files,
            GlobalFileManager._search_files,
            include_class_name=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# WebSearcher mirroring
# ─────────────────────────────────────────────────────────────────────────────


def mirror_web_searcher_tools() -> Dict[str, Any]:
    """Build a tool-dict mirroring the real WebSearcher's ask tools.

    Uses AST reflection of WebSearcher.__init__ with a static fallback.
    """
    from unity.common.llm_helpers import methods_to_tool_dict
    from unity.web_searcher.web_searcher import WebSearcher

    try:
        pairs = _extract_owner_method_pairs(
            WebSearcher,
            "_ask_tools",
            self_external_map=None,
            extra_class_names={"WebSearcher": WebSearcher},
        )
        if pairs:
            tools = _build_tool_dict(pairs)
            if tools:
                return tools
    except Exception:
        pass

    # Fallback – current canonical ask tool set
    return methods_to_tool_dict(
        WebSearcher._search,
        WebSearcher._extract,
        WebSearcher._crawl,
        WebSearcher._map,
        include_class_name=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GuidanceManager mirroring
# ─────────────────────────────────────────────────────────────────────────────


def mirror_guidance_manager_tools(kind: str) -> Dict[str, Any]:
    """Build a tool-dict mirroring the real GuidanceManager's tools.

    kind: "ask" or "update". Uses AST reflection of GuidanceManager.__init__
    with a static fallback kept in sync with the concrete implementation.
    """
    from unity.common.llm_helpers import methods_to_tool_dict
    from unity.guidance_manager.guidance_manager import GuidanceManager as GM

    target_attr = "_ask_tools" if kind == "ask" else "_update_tools"

    try:
        pairs = _extract_owner_method_pairs(
            GM,
            target_attr,
            self_external_map=None,
            extra_class_names={"GuidanceManager": GM},
        )
        if pairs:
            tools = _build_tool_dict(pairs)
            if tools:
                return tools
    except Exception:
        pass

    # Fallback – keep aligned with GuidanceManager.__init__
    if kind == "ask":
        return methods_to_tool_dict(
            GM._list_columns,
            GM._filter,
            GM._search,
            GM._get_images_for_guidance,
            GM._ask_image,
            GM._attach_image_to_context,
            GM._attach_guidance_images_to_context,
            GM._get_functions_for_guidance,
            GM._attach_functions_for_guidance_to_context,
            include_class_name=False,
        )
    else:
        return methods_to_tool_dict(
            GM.ask,
            GM._add_guidance,
            GM._update_guidance,
            GM._delete_guidance,
            GM._create_custom_column,
            GM._delete_custom_column,
            include_class_name=False,
        )
