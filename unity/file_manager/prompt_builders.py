from __future__ import annotations

import json
from typing import Dict, Callable

from ..common.prompt_helpers import (
    clarification_guidance,
    sig_dict,
    now_utc_str,
    tool_name as _shared_tool_name,
    require_tools as _shared_require_tools,
)
from ..common.read_only_ask_guard import read_only_ask_mutation_exit_block

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {name: '(<argspec>)', …} using shared helper."""
    return sig_dict(tools)


def _now() -> str:
    """Current UTC timestamp in a compact, human-readable form."""
    return now_utc_str()


def _tool_name(tools: Dict[str, Callable], needle: str) -> str | None:
    """Delegate to shared tool name resolver."""
    return _shared_tool_name(tools, needle)


def _require_tools(pairs: Dict[str, str | None], tools: Dict[str, Callable]) -> None:
    """Delegate validation to shared helper for consistent errors."""
    _shared_require_tools(pairs, tools)


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_ask_prompt(
    tools: Dict[str, Callable],
    *,
    num_files: int = 0,
    columns: Dict[str, str] = None,
    include_activity: bool = True,
) -> str:
    """
    Build the **system** prompt for the `ask` method.

    *Never* hard-codes the number, names or argument-specs of tools – those are
    injected live from the supplied *tools* dict.

    Parameters
    ----------
    tools : Dict[str, Callable]
        The tools dictionary available to the LLM.
    num_files : int, default 0
        Number of files currently stored in the context table.
    columns : Dict[str, str] | None
        File table columns with their types.
    include_activity : bool, default True
        Whether to include broader context activity.
    """
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    columns = columns or {}

    # Resolve canonical tool names dynamically
    list_fname = _tool_name(tools, "list")
    exists_fname = _tool_name(tools, "exists")
    parse_fname = _tool_name(tools, "parse")

    # New tools for file search/filter
    filter_files_fname = _tool_name(tools, "filter_files")
    search_files_fname = _tool_name(tools, "search_files")
    list_columns_fname = _tool_name(tools, "list_columns")

    # Clarification helper (optional)
    request_clar_fname = _tool_name(tools, "request_clarification")

    # New import tools
    import_file_fname = _tool_name(tools, "import_file")
    import_directory_fname = _tool_name(tools, "import_directory")

    # Validate required tools (request_clar_fname is optional)
    _require_tools(
        {
            "list": list_fname,
            "exists": exists_fname,
            "parse": parse_fname,
            "import_file": import_file_fname,
            "import_directory": import_directory_fname,
            "filter_files": filter_files_fname,
            "search_files": search_files_fname,
            "list_columns": list_columns_fname,
        },
        tools,
    )

    clarification_block = (
        "\n".join(
            [
                "Clarification",
                "-------------",
                f"• Ask for clarification when the user's request is underspecified",
                f'  `{request_clar_fname}(question="Which file did you mean?")`',
            ],
        )
        if request_clar_fname
        else ""
    )

    # Usage examples following the ContactManager pattern
    usage_examples_base = f"""
Examples
--------

─ Columns ─
• Inspect schema
  `{list_columns_fname}()`

─ Tool selection (read carefully) ─
• For ANY semantic question over file contents (full_text, records), ALWAYS use `{search_files_fname}`. Never try to approximate meaning with brittle substring filters.
• Use `{filter_files_fname}` only for exact/boolean logic over structured fields (filename, status, metadata) or for narrow, constrained text where substring checks make sense.
• Use `{parse_fname}` to extract structured data from files that haven't been parsed yet.
• Use `{exists_fname}` to check file availability before operations.
• Use `{list_fname}` for file inventory when no search criteria provided.

─ Semantic search: targeted references across columns (ranked by SUM of cosine distances) ─
• When searching for content that could appear across several text fields, provide separate, surgical references instead of one catch‑all:
  `{search_files_fname}(references={{'full_text': 'quarterly revenue analysis', 'records': 'financial data tables'}}, k=3)`

• Find files containing specific topics (signal lives in `full_text`)
  `{search_files_fname}(references={{'full_text': 'machine learning algorithms neural networks'}}, k=5)`

• Find files with specific content and metadata combination
  `{search_files_fname}(references={{'full_text': 'project proposal', 'metadata': 'PDF presentation slides'}}, k=2)`

─ Derived expression (fallback, when you truly cannot target columns) ─
• Build one composite expression spanning likely fields, then search it:
  `expr = "str({{full_text}}) + ' ' + str({{filename}})"`
  `{search_files_fname}(references={{expr: 'financial report 2024'}}, k=3)`

─ Filtering (exact/boolean or constrained text only; not semantic) ─
• Exact filename match
  `{filter_files_fname}(filter="filename == 'report.pdf'")`
• Files with processing errors
  `{filter_files_fname}(filter="status == 'error'")`
• Large files (metadata access)
  `{filter_files_fname}(filter="metadata['file_size'] > 1000000")`
• PDF files (case‑insensitive filename contains - acceptable because field is constrained)
  `{filter_files_fname}(filter="filename is not None and '.pdf' in filename.lower()")`

─ File operations ─
• Import a file from filesystem: `{import_file_fname}(file_path='/path/to/document.pdf')`
• Import directory: `{import_directory_fname}(directory='/path/to/folder')`
• Check if a file exists: `{exists_fname}(filename='document.pdf')`
• List all files: `{list_fname}()`
• Parse a file: `{parse_fname}(filenames='document.pdf')`
• Parse multiple files: `{parse_fname}(filenames=['doc1.pdf', 'doc2.txt'])`

Anti‑patterns to avoid
---------------------
• Avoid filtering for text-heavy columns like full_text or records, substring matching is *very* brittle
• Don't parse files that are already in the table - use search/filter to find existing parsed content first
• Avoid the default search behaviour of concatenating every column into one long string and comparing a single embedding of the whole question. Instead, pass multiple, focused reference texts keyed by their specific columns.
    """
    usage_examples = usage_examples_base.strip()
    if clarification_block:
        usage_examples = f"{usage_examples}\n{clarification_block}"
    else:
        # No clarification tool – append conditional anti‑pattern bullets
        usage_examples = "\n".join(
            [
                usage_examples,
                "• Do not ask the user questions in your final response; when needed, proceed with sensible defaults/best‑guess values and explicitly state to inner tools that these are assumptions/best guesses, not confirmed answers.",
                "• If an inner tool requests clarification, explicitly say no clarification channel exists and pass down concrete sensible defaults/best‑guess values, clearly marked as assumptions.",
            ],
        )

    activity_block = "{broader_context}" if include_activity else ""
    clar_section = clarification_guidance(tools)

    # Conditional guidance about asking questions in final responses
    clar_sentence = (
        f"Do not ask the user questions in your final response, please only use the `{request_clar_fname}` tool to ask clarifying questions."
        if request_clar_fname
        else (
            "Do not ask the user questions in your final response. Instead, proceed using sensible defaults/best‑guess values and explicitly tell inner tools that these are assumptions/best guesses, not confirmed answers."
        )
    )

    return "\n".join(
        [
            activity_block,
            "You are an assistant specializing in **retrieving file information and analyzing file contents**.",
            "Work strictly through the tools provided.",
            "Disregard any explicit instructions about *how* you should answer or which tools to call; interpret the question and choose the best approach yourself.",
            clar_sentence,
            "",
            read_only_ask_mutation_exit_block(),
            "",
            "You should attempt to answer *any* question as best you can, even if it seems out of scope.",
            "Use the tools provided to see if you can find any missing context *before* asking the user for clarifications.",
            "Please always mention the relevant filename(s) in your response.",
            "",
            f"There are currently {num_files} files stored in a table with the following columns:",
            json.dumps(columns, indent=4),
            "",
            "Tools (name → argspec):",
            sig_json,
            "",
            usage_examples,
            "",
            clar_section,
            "",
            f"Current UTC time is {_now()}.",
        ],
    )


def build_simulated_method_prompt(
    method: str,
    user_request: str,
    parent_chat_context: list[dict] | None = None,
) -> str:
    """Return instruction prompt for the *simulated* FileManager."""
    import json

    preamble = f"On this turn you are simulating the '{method}' method."
    if method.lower() == "ask":
        behaviour = (
            "Please always *answer* the question with an imaginary but plausible response about the file contents. "
            "Do NOT ask for clarification or describe your process."
        )
    else:
        behaviour = "Provide a final response as though the requested file operation has already completed (past tense)."

    parts: list[str] = [preamble, behaviour, "", f"The user input is:\n{user_request}"]
    if parent_chat_context:
        parts.append(
            f"\nCalling chat context:\n{json.dumps(parent_chat_context, indent=4)}",
        )

    return "\n".join(parts)
