from __future__ import annotations

import json
import textwrap
from typing import Callable, Dict, List

from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now_utc_str,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
)
from ..common.read_only_ask_guard import read_only_ask_mutation_exit_block


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    return sig_dict(tools)


def _tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    return _shared_tool_name(tools, needle)


def _require_tools(pairs: Dict[str, str | None], tools: Dict[str, Callable]) -> None:
    _shared_require_tools(pairs, tools)


def _now() -> str:
    return now_utc_str()


def build_ask_prompt(
    tools: Dict[str, Callable],
    *,
    include_activity: bool = True,
    num_functions: int | None = None,
    function_columns: List[Dict[str, str]] | Dict[str, str] | List[str] | None = None,
) -> str:
    """
    Build the system prompt for SkillManager.ask.

    The SkillManager answers questions about the assistant's high-level skills
    by consulting the read-only tools exposed by FunctionManager (list/search/similarity).
    """

    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Resolve canonical names dynamically (optional presence)
    list_fname = _tool_name(tools, "list_functions")
    search_fname = _tool_name(tools, "search_functions")
    similar_fname = _tool_name(tools, "search_functions_by_similarity")
    get_prec_fname = _tool_name(tools, "get_precondition")
    request_clar_fname = _tool_name(tools, "request_clarification")

    _require_tools(
        {
            "list_functions": list_fname,
            "search_functions": search_fname,
            "search_functions_by_similarity": similar_fname,
        },
        tools,
    )

    clarification_block = (
        textwrap.dedent(
            f"""
            Clarification
            -------------
            • If the request is ambiguous (e.g., "show skills for data"), ask the user to refine the domain or goal first:
              `{request_clar_fname}(question="Which skill domain? e.g., spreadsheets, web, audio, data-cleaning?")`
            """,
        ).strip()
        if request_clar_fname
        else ""
    )

    usage_examples = textwrap.dedent(
        f"""
        Examples
        --------

        ─ Catalogue overview ─
        • List all available skills (functions with short descriptions)
          `{list_fname}()`

        ─ Search by keywords ─
        • Find skills related to spreadsheets and CSVs
          `{search_fname}(filter="'csv' in docstring or 'spreadsheet' in docstring", limit=10)`

        ─ Semantic similarity ─
        • Ask for skills similar to a natural-language intent
          `{similar_fname}(query="convert spreadsheet to CSV and clean duplicates", n=5)`

        ─ Preconditions / capabilities ─
        • Check if a skill requires external configuration
          `{get_prec_fname}(function_name="fetch_webpage")`

        Anti‑patterns to avoid
        ----------------------
        • Do not expose raw function names or signatures as the final skill label – translate into human, anthropomorphic capabilities.
        • Do not present code blocks or argspecs unless the user explicitly asks for underlying functions.
        • Do not invent function signatures – consult the tools and quote real `argspec`.
        • Avoid listing raw implementations unless explicitly asked; prefer names, signatures and docstrings.
        • Do not call mutation endpoints (adding/deleting functions) from SkillManager.ask – it is read‑only.
        """,
    ).strip()

    activity_block = "{broader_context}" if include_activity else ""
    clar_sentence = (
        f"Do not ask the user questions in your final response, please only use the `{request_clar_fname}` tool to ask clarifying questions."
        if request_clar_fname
        else (
            "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."
        )
    )
    clar_section = clarification_guidance(tools)

    # CRITICAL phrasing guidance: translate functions → human skills
    phrasing_block = textwrap.dedent(
        """
        Skill phrasing (CRITICAL)
        ------------------------
        • Present answers as high‑level, anthropomorphic skills, not code.
        • Translate function names and docstrings into natural skill descriptors.
        • Default: do not show raw function names or signatures.
        • Only include the underlying function name in parentheses if explicitly requested by the user.

        Examples of phrasing
        • add_integers() → "good at mental arithmetic"
        • search_google() → "experienced at web browsing and finding information quickly"
        • summarise_text() → "skilled at concise summarisation of long documents"
        """,
    ).strip()

    counts_block = (
        f"There are currently {num_functions} stored functions (skills catalogue)."
        if isinstance(num_functions, int)
        else ""
    )

    columns_block = (
        "Function columns (from Function schema):\n"
        + json.dumps(function_columns, indent=4)
        if function_columns is not None
        else ""
    )

    return "\n".join(
        [
            activity_block,
            "You are an assistant specialised in describing the assistant's high‑level skills.",
            "Work strictly through the tools provided to discover skills and their details.",
            clar_sentence,
            phrasing_block,
            counts_block,
            columns_block,
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            usage_examples,
            "",
            clar_section,
            "",
            "",
            read_only_ask_mutation_exit_block(),
            "",
            f"Current UTC time is {_now()}.",
        ],
    )
