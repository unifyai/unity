import json

import pytest
import unify

from unity.function_manager.function_manager import FunctionManager
from tests.helpers import _handle_project


@pytest.mark.requires_real_unify
@_handle_project
def test_embedding_and_similarity_search():
    """
    Tests that function embeddings are created and that semantic search
    can find functions by similarity.
    """
    fm = FunctionManager()

    # 1. Add functions with distinct semantic meanings
    pay_bill_src = """
def pay_utility_bill_via_console(vendor: str, amount: float):
        \"\"\"Pays a utility bill through the command line interface.\"\"\"
        return f"Paying {amount} to {vendor} via console."
"""

    search_docs_src = """
def search_internal_documents(query: str):
        \"\"\"Searches the internal knowledge base for a given query.\"\"\"
        return f"Searching for {query} in internal documents."
"""

    send_email_src = """
def send_plain_text_email(recipient: str, subject: str, body: str):
        \"\"\"Sends a simple email with a plain text body.\"\"\"
        return f"Sending email to {recipient} with subject '{subject}'."
"""

    fm.add_functions(implementations=[pay_bill_src, search_docs_src, send_email_src])

    # 2. Check that a keyword search for a non-existent term fails
    keyword_hits = fm.filter_functions(filter="'gas' in docstring")
    assert not keyword_hits

    # 3. Use semantic search to find the most similar function
    query = "pay gas bill online"
    similar_funcs = fm.search_functions(query=query, n=1)

    assert len(similar_funcs) == 1
    assert similar_funcs[0]["name"] == "pay_utility_bill_via_console"

    # 4. Test the 'n' parameter
    similar_funcs_2 = fm.search_functions(query=query, n=2)
    assert len(similar_funcs_2) == 2
    assert similar_funcs_2[0]["name"] == "pay_utility_bill_via_console"


def _get_embedding(fm: FunctionManager, name: str):
    """Fetch the raw _embedding_text_emb for a function by name."""
    logs = unify.get_logs(
        context=fm._compositional_ctx,
        filter=f"name == {json.dumps(name)}",
        limit=1,
    )
    assert logs, f"No log found for {name!r}"
    return logs[0].entries.get("_embedding_text_emb"), logs[0]


@pytest.mark.requires_real_unify
@_handle_project
def test_embedding_populated_on_insert():
    """After add_functions + warm_embeddings (production startup order), embeddings exist."""
    fm = FunctionManager()
    fm.add_functions(
        implementations=[
            """
def invoice_reconciliation_tool(ledger_id: str) -> str:
    \"\"\"Reconcile supplier invoices against the general ledger.\"\"\"
    return ledger_id
""",
        ],
    )
    fm.warm_embeddings()
    emb, log = _get_embedding(fm, "invoice_reconciliation_tool")
    assert emb is not None, (
        f"_embedding_text_emb is None for log.id={log.id}; "
        f"embedding_text={log.entries.get('embedding_text')!r}"
    )
    assert isinstance(emb, list) and len(emb) > 0


@pytest.mark.requires_real_unify
@_handle_project
def test_embedding_refreshed_on_overwrite():
    """After overwriting a function, the row has a fresh embedding via update ripple."""
    fm = FunctionManager()
    fm.add_functions(
        implementations=[
            """
def overwrite_target_fn() -> str:
    \"\"\"Original: bake sourdough bread loaves for the bakery.\"\"\"
    return "v1"
""",
        ],
    )
    fm.warm_embeddings()
    original_emb, _ = _get_embedding(fm, "overwrite_target_fn")
    assert original_emb is not None, "_embedding_text_emb missing after insert"

    fm.add_functions(
        implementations=[
            """
def overwrite_target_fn() -> str:
    \"\"\"Updated: reconcile quarterly expenses against budget forecasts.\"\"\"
    return "v2"
""",
        ],
        overwrite=True,
    )
    updated_emb, _ = _get_embedding(fm, "overwrite_target_fn")
    assert updated_emb is not None, "_embedding_text_emb missing after overwrite"
    assert isinstance(updated_emb, list) and len(updated_emb) > 0
    assert (
        updated_emb != original_emb
    ), "Embedding vector should differ after semantically different overwrite"
