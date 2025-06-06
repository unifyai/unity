from __future__ import annotations

import inspect
import json
import textwrap
from datetime import datetime, timezone
from typing import Dict, Callable

from .types.contact import Contact
from ..common.llm_helpers import SteerableToolHandle, class_api_overview


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sig_dict(tools: Dict[str, Callable]) -> Dict[str, str]:
    """Return {tool_name: '(<argspec>)', …} for pretty JSON dumps."""
    return {n: str(inspect.signature(fn)) for n, fn in tools.items()}


def _now() -> str:  # UTC timestamp helper
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_ask_prompt(tools: Dict[str, Callable]) -> str:
    """Return the system-prompt used by *ask*."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)
    # Assume there is exactly *one* search-tool in the dict:
    search_name = next(iter(tools))

    usage_examples = textwrap.dedent(
        f"""
        Examples
        --------
        • Find contacts with first name **John**
          `{search_name}(filter="first_name == 'John'")`
        • Find surname **Doe**
          `{search_name}(filter="surname == 'Doe'")`
        • Specific email **john.doe@example.com**
          `{search_name}(filter="email_address == 'john.doe@example.com'")`
        • Phone containing **555**
          `{search_name}(filter="'555' in phone_number")`
        • Exact phone **+14445556666**
          `{search_name}(filter="phone_number == '+14445556666'")`
        • Name **Alice Smith**
          `{search_name}(filter="surname == 'Smith' and first_name == 'Alice'")`
        • Email **a@b.com** *or* phone **123-456-7890**
          `{search_name}(filter="email_address == 'a@b.com' or phone_number == '123-456-7890'")`
        • Missing phone number
          `{search_name}(filter="phone_number is None")`
        • Has any email (not None)
          `{search_name}(filter="email_address is not None")`
    """,
    ).strip()

    return textwrap.dedent(
        f"""
        You are an assistant specializing in **retrieving contact information**.
        Work strictly through the tools provided.

        Tools (name → argspec):
        {sig_json}

        {usage_examples}

        Contact schema:
        {json.dumps(Contact.model_json_schema(), indent=4)}

        SteerableToolHandle class:
        {class_api_overview(SteerableToolHandle)}

        Current UTC time is {_now()}.
    """,
    ).strip()


def build_update_prompt(tools: Dict[str, Callable]) -> str:
    """Return the system-prompt used by *update*."""
    sig_json = json.dumps(_sig_dict(tools), indent=4)

    # Pick out canonical names heuristically
    create_name = next((n for n in tools if "create" in n.lower()), None)
    update_name = next((n for n in tools if "update" in n.lower()), None)
    search_name = next((n for n in tools if "search" in n.lower()), None)

    usage_examples = textwrap.dedent(
        f"""
        Examples
        --------
        • **Create** a new contact
          `{create_name}(first_name='Jane', surname='Roe', email_address='jane.roe@example.com')`

        • **Update** John Doe's phone '+1 55512-345-67' when you already know the ID is *42*
          `{update_name}(contact_id=42, phone_number='+15551234567')` (note spaces and dashes removed)

        • **Update** a contact referred to only by name
          1 Find ID → `{search_name}(filter="first_name == 'John' and surname == 'Doe'")`
          2 Then update → `{update_name}(contact_id=<returned_id>, email_address='john.new@example.com')`

        • **Parse** a full name on create
          `"Frank P. Castle"` → `{create_name}(first_name='Frank P.', surname='Castle')`
    """,
    ).strip()

    return textwrap.dedent(
        f"""
        You are an assistant in charge of **creating or editing contacts**.
        Use the tools provided to create new entries or update existing ones.

        Tools (name → argspec):
        {sig_json}

        {usage_examples}

        Contact schema:
        {json.dumps(Contact.model_json_schema(), indent=4)}

        SteerableToolHandle class:
        {class_api_overview(SteerableToolHandle)}

        Current UTC time is {_now()}.
    """,
    ).strip()
