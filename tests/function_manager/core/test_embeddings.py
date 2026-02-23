import pytest
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
