from __future__ import annotations

"""
RepairsAgent: Minimal facade over LocalFileManager.ask with BusinessContextPayload.

This facade prepares a structured BusinessContextPayload that is injected into the
FileManager.ask system prompt via slot-filling. It encapsulates:
  - role_description: Primary identity (data analyst for Midland Heart)
  - domain_rules: Data sources, schemas, join logic, business rules
  - response_guidelines: Citation format, confidence scores, output style
  - retrieval_hints: Domain-specific query patterns

Sandbox mode:
  - If sandbox_mode=True, ask(...) returns the SteerableToolHandle directly.
  - Otherwise, it awaits the handle.result() and returns the final result payload.
"""

import asyncio
from typing import Any, Dict, List, Optional

import unify
from .prompt_builder import build_repairs_ask_instructions
from unity.file_manager.managers.local import LocalFileManager as FileManager
from unity.file_manager.types.config import FilePipelineConfig


# Pipeline config containing business contexts (file paths, table/column descriptions, rules)
PIPELINE_CONFIG_PATH = (
    "/home/hmahmood24/unity/intranet/repairs/repairs_file_pipeline_config.json"
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
        Ask a repairs data question with BusinessContextPayload for slot-filling.
        """
        business_data = self._build_business_payload()
        business_payload = build_repairs_ask_instructions(business_data)

        # TODO: REMOVE - Debug file dumps for prompt inspection
        import json

        json.dump(business_data, open("business_payload.json", "w"), indent=4)

        handle = await self.file_manager.ask(  # type: ignore
            question,
            _return_reasoning_steps=_return_reasoning_steps,
            _parent_chat_context=_parent_chat_context,
            _clarification_up_q=_clarification_up_q,
            _clarification_down_q=_clarification_down_q,
            rolling_summary_in_prompts=rolling_summary_in_prompts,
            business_payload=business_payload,
        )
        if self.sandbox_mode:
            return handle
        try:
            return await handle.result()
        except Exception:
            return {"error": "Failed to obtain result from handle"}

    def _build_business_payload(self) -> Dict[str, Any]:
        """
        Collect schema, searchable columns, samples, and rules for all repairs files.
        Best-effort: any failures return partial data.

        Returns a payload structured with:
        - global_rules: List[str] - cross-file rules
        - files: Dict[file_path, file_payload] where file_payload contains:
            - file_rules: List[str] - rules for this file
            - tables: Dict[table_label, table_payload] where table_payload contains:
                - table_rules: List[str] - rules for this table
                - schema: List[Dict] - column info
                - searchable_columns: List[str]
                - samples: List[Dict]
        """
        config_data = self._load_config_data()
        fallback_descriptions = config_data["fallback_descriptions"]
        fallback_table_rules = config_data["fallback_table_rules"]
        file_paths = config_data["file_paths"]
        file_rules_map = config_data["file_rules"]
        global_rules = config_data["global_rules"]

        # Build per-file payloads
        files_payload: Dict[str, Any] = {}

        for file_path in file_paths:
            file_payload = self._build_file_payload(
                file_path,
                fallback_descriptions,
                fallback_table_rules,
                file_rules_map.get(file_path, []),
            )
            if file_payload:
                files_payload[file_path] = file_payload

        return {
            "global_rules": global_rules,
            "files": files_payload,
        }

    def _load_config_data(self) -> Dict[str, Any]:
        """
        Load file paths, column descriptions, and all rules from the pipeline config JSON.

        Returns:
            Dict with keys:
                - file_paths: list of file paths from business_contexts.file_contexts
                - fallback_descriptions: nested dict file_path → table_label → column_name → description
                - fallback_table_rules: nested dict file_path → table_label → List[str]
                - file_rules: dict file_path → List[str]
                - global_rules: List[str]
        """
        file_paths: List[str] = []
        fallback: Dict[str, Dict[str, Dict[str, str]]] = {}
        fallback_table_rules: Dict[str, Dict[str, List[str]]] = {}
        file_rules: Dict[str, List[str]] = {}
        global_rules: List[str] = []

        try:
            config = FilePipelineConfig.from_file(PIPELINE_CONFIG_PATH)
            business_contexts = config.ingest.business_contexts
            if business_contexts:
                # Extract global rules
                global_rules = list(business_contexts.global_rules or [])

                # Extract file-level data
                for fc in business_contexts.file_contexts:
                    file_paths.append(fc.file_path)
                    file_rules[fc.file_path] = list(fc.file_rules or [])

                    file_fallback: Dict[str, Dict[str, str]] = {}
                    file_table_rules: Dict[str, List[str]] = {}

                    for table_spec in fc.table_contexts:
                        file_fallback[table_spec.table] = dict(
                            table_spec.column_descriptions,
                        )
                        file_table_rules[table_spec.table] = list(
                            table_spec.table_rules or [],
                        )

                    fallback[fc.file_path] = file_fallback
                    fallback_table_rules[fc.file_path] = file_table_rules
        except Exception:
            pass

        return {
            "file_paths": file_paths,
            "fallback_descriptions": fallback,
            "fallback_table_rules": fallback_table_rules,
            "file_rules": file_rules,
            "global_rules": global_rules,
        }

    def _build_file_payload(
        self,
        file_path: str,
        fallback_descriptions: Dict[str, Dict[str, Dict[str, str]]],
        fallback_table_rules: Dict[str, Dict[str, List[str]]],
        file_rules: List[str],
    ) -> Dict[str, Any]:
        """
        Build schema, searchable columns, samples, and rules for a single file.
        """
        # Get tables overview for this file
        try:
            overview = self.file_manager._tables_overview(file=file_path)  # type: ignore[attr-defined]
        except Exception:
            return {}

        # Extract table contexts from overview
        table_contexts: Dict[str, str] = {}  # table_label → context_string
        for key, val in (overview or {}).items():
            if key == "FileRecords" or not isinstance(val, dict):
                continue
            tables = val.get("Tables")
            if not isinstance(tables, dict):
                continue
            for table_label, info in tables.items():
                ctx = (info or {}).get("context")
                if isinstance(ctx, str) and ctx:
                    table_contexts[table_label] = ctx

        if not table_contexts:
            return {}

        # Get fallback descriptions and rules for this file
        file_fallback = fallback_descriptions.get(file_path, {})
        file_table_rules = fallback_table_rules.get(file_path, {})

        # Build per-table payloads
        tables_payload: Dict[str, Any] = {}
        all_searchable_cols: List[str] = []

        for table_label, ctx in table_contexts.items():
            table_payload = self._build_table_payload(
                ctx=ctx,
                table_label=table_label,
                fallback_col_descriptions=file_fallback.get(table_label, {}),
                table_rules=file_table_rules.get(table_label, []),
            )
            if table_payload:
                tables_payload[table_label] = table_payload
                all_searchable_cols.extend(table_payload.get("searchable_columns", []))

        return {
            "file_rules": file_rules,
            "tables": tables_payload,
            "searchable_columns": list(set(all_searchable_cols)),
        }

    def _build_table_payload(
        self,
        ctx: str,
        table_label: str,
        fallback_col_descriptions: Dict[str, str],
        table_rules: List[str],
    ) -> Dict[str, Any]:
        """
        Build schema, searchable columns, samples, and rules for a single table context.
        """
        # Get fields from Unify
        fields: Dict[str, Any] = {}
        try:
            fields = unify.get_fields(context=ctx) or {}
        except Exception:
            fields = {}

        # Track private fields to exclude from samples
        exclude_fields = [
            fname
            for fname in list(fields.keys())
            if isinstance(fname, str) and fname.startswith("_")
        ]

        # Derive searchable columns from _X_emb fields
        searchable_cols: List[str] = []
        for fname in list(fields.keys()):
            if (
                isinstance(fname, str)
                and fname.startswith("_")
                and fname.endswith("_emb")
            ):
                col = fname[1:-4]  # drop leading '_' and trailing '_emb'
                if col and col not in searchable_cols:
                    searchable_cols.append(col)

        # Build schema with descriptions
        # Priority: 1) description from unify.get_fields, 2) fallback from config JSON
        schema_rows: List[Dict[str, Any]] = []
        for fname, field_info in fields.items():
            if not isinstance(fname, str):
                continue
            if fname.startswith("_") and fname.endswith("_emb"):
                continue  # skip vector columns from schema list

            # Try to get description from field_info first
            description = None
            if isinstance(field_info, dict):
                description = field_info.get("description")

            # TODO: Remove this fallback once RepairsAgent project has finished ingestion
            # and all column descriptions are stored in Unify fields directly.
            if not description:
                description = fallback_col_descriptions.get(fname)

            schema_rows.append(
                {
                    "name": fname,
                    "description": description,
                },
            )

        # Fetch sample rows
        samples: List[Dict[str, Any]] = []
        try:
            logs = unify.get_logs(
                context=ctx,
                limit=3,
                exclude_fields=exclude_fields,
            )
            for lg in logs or []:
                entry = getattr(lg, "entries", None)
                if isinstance(entry, dict):
                    samples.append(entry)
        except Exception:
            pass

        return {
            "table_rules": table_rules,
            "schema": schema_rows,
            "searchable_columns": searchable_cols,
            "samples": samples,
        }
