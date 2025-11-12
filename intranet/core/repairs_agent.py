from __future__ import annotations

"""
RepairsAgent: Minimal facade over LocalFileManager.ask with business context.

This facade prepares a compact "business_context" string that is appended to the
FileManager.ask system prompt. It encapsulates:
  - schema (name, data_type, optional description) for each table
  - pre-embedded semantic search targets (derived from _X_emb fields)
  - small samples (up to 10 entries per context)
  - domain/business rules for Midland Heart repairs data

Sandbox mode:
  - If sandbox_mode=True, ask(...) returns the SteerableToolHandle directly.
  - Otherwise, it awaits the handle.result() and returns the final result payload.
"""

import asyncio
from typing import Any, Dict, List, Optional

import unify
from .prompt_builder import build_repairs_ask_instructions
from unity.file_manager.managers.local import LocalFileManager as FileManager


REPAIRS_FILE_PATH = "/home/hmahmood24/unity/intranet/repairs/MDH Repairs data July & Aug 25 - DL V1.xlsx"
SCHEMA_TEMPLATE_PATH = (
    "/home/hmahmood24/unity/intranet/repairs/column_descriptions.json"
)


class RepairsAgent:
    def __init__(self, sandbox_mode: bool = False) -> None:
        """
        Construct the RepairsAgent with a LocalFileManager.

        Parameters
        ----------
        sandbox_mode : bool
            When True, ask(...) returns a SteerableToolHandle; otherwise returns awaited result.
        """
        self.sandbox_mode = sandbox_mode
        self.file_manager = FileManager()

    async def ask(
        self,
        question: str,
        _return_reasoning_steps: bool = False,
        _parent_chat_context: list[dict] | None = None,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
        rolling_summary_in_prompts: Optional[bool] = None,
    ) -> Any:
        """
        Ask a repairs data question with business-specific context appended.
        """
        business_payload = self._build_business_payload()
        business_context = build_repairs_ask_instructions(business_payload)

        import json

        json.dump(business_payload, open("business_payload.json", "w"), indent=4)
        open("business_context.txt", "w").write(business_context)

        handle = await self.file_manager.ask(  # type: ignore
            question,
            _return_reasoning_steps=_return_reasoning_steps,
            _parent_chat_context=_parent_chat_context,
            _clarification_up_q=_clarification_up_q,
            _clarification_down_q=_clarification_down_q,
            rolling_summary_in_prompts=rolling_summary_in_prompts,
            business_context=business_context,
        )
        if self.sandbox_mode:
            return handle
        try:
            return await handle.result()
        except Exception:
            return {"error": "Failed to obtain result from handle"}

    def _build_business_payload(self) -> Dict[str, Any]:
        """
        Collect schema, searchable columns, and small samples for the repairs file.
        Best-effort: any failures return partial data.
        """
        # 1) Tables overview (file-scoped)
        try:
            overview = self.file_manager._tables_overview(file=REPAIRS_FILE_PATH)  # type: ignore[attr-defined]
        except Exception:
            overview = {}

        import json

        json.dump(overview, open("overview.json", "w"), indent=4)

        table_contexts: List[str] = []
        for key, val in (overview or {}).items():
            if key == "FileRecords" or not isinstance(val, dict):
                continue
            tables = val.get("Tables")
            if not isinstance(tables, dict):
                continue
            for _, info in tables.items():
                ctx = (info or {}).get("context")
                if isinstance(ctx, str) and ctx:
                    table_contexts.append(ctx)

        # 2) Load column descriptions template (optional)
        descriptions: Dict[str, str] = {}
        try:
            with open(SCHEMA_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                tmpl = json.load(f)
            descriptions = dict(tmpl.get("columns", {}))
            business_rules = list(tmpl.get("business_rules", []))
        except Exception:
            business_rules = [
                "Jobs completed: distinct jobs where WorksOrderStatusDescription ∈ {Complete, Closed}.",
                "WorksOrderStatusDescription: Complete = operative finished works; Closed = financials completed; typical flow Complete → Closed.",
                "Planned window: JobTicketLinePlannedStartDate/EndDate set by scheduling team.",
                "WorksOrderReportedCompletedDate: when all job tickets on the works order are completed.",
                "ArrivedOnSite: operative’s tablet ‘arrived’ click; CompletedVisit: operative pressed complete. Some may click through fast (close times).",
                "VisitDate: when the operative accepts the job ticket on their tablet.",
                "Deduplication: prefer JobTicketReference for removing duplicates; WorksOrderRef duplicates may be valid follow‑ons.",
            ]

        # 3) Build schema and searchable columns
        schema_rows: List[Dict[str, Any]] = []
        searchable_cols: List[str] = []
        exclude_fields_by_ctx: Dict[str, List[str]] = {}
        for ctx in table_contexts:
            fields = {}
            try:
                fields = unify.get_fields(context=ctx) or {}
            except Exception:
                fields = {}
            # Track private fields (leading "_") to exclude from example logs
            exclude_fields_by_ctx[ctx] = [
                fname
                for fname in list(fields.keys())
                if isinstance(fname, str) and fname.startswith("_")
            ]
            # Derive searchable → any field named `_X_emb` → `X`
            for fname in list(fields.keys()):
                if (
                    isinstance(fname, str)
                    and fname.startswith("_")
                    and fname.endswith("_emb")
                ):
                    col = fname[1:-4]  # drop leading '_' and trailing '_emb'
                    if col and col not in searchable_cols:
                        searchable_cols.append(col)
            # Add non-embedded schema entries
            for fname, _ in fields.items():
                if not isinstance(fname, str):
                    continue
                if fname.startswith("_") and fname.endswith("_emb"):
                    continue  # skip vector columns from schema list
                schema_rows.append(
                    {
                        "name": fname,
                        "description": descriptions.get(fname),
                    },
                )

        # 4) Samples: up to 10 entries per context
        samples_by_ctx: Dict[str, Any] = {}
        for ctx in table_contexts:
            try:
                exclude_private = exclude_fields_by_ctx.get(ctx) or []
                logs = unify.get_logs(
                    context=ctx,
                    limit=3,
                    exclude_fields=exclude_private,
                )
                rows = []
                for lg in logs or []:
                    entry = getattr(lg, "entries", None)
                    if isinstance(entry, dict):
                        rows.append(entry)
                samples_by_ctx[ctx] = rows
            except Exception:
                samples_by_ctx[ctx] = []

        return {
            "schema": schema_rows,
            "searchable_columns": searchable_cols,
            "samples": samples_by_ctx,
            "business_rules": business_rules,
        }
