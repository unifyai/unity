"""
Vector-embedding & semantic-search tests for **ContactManager**.

We mirror the pattern used in `test_knowledge_embedding` but operate on the
single contacts table, exercising `_nearest_column` and the automatic creation
of `<source>_vec` derived columns.
"""

import pytest

from unity.contact_manager.contact_manager import ContactManager

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_contact_embedding_and_nearest_search():
    cm = ContactManager()

    # ------------------------------------------------------------------ #
    # 1️⃣  Create three contacts whose *description* fields are related  #
    #     without sharing obvious substrings.                            #
    # ------------------------------------------------------------------ #
    entries = [
        ("Alice", "I email and phone sometimes."),
        ("Bob", "Text messaging is my go-to communication method."),
        ("Carol", "I love taking the train to work."),
    ]
    for fname, desc in entries:
        cm._create_contact(first_name=fname, description=desc)

    # ------------------------------------------------------------------ #
    # 2️⃣  Keyword search for a term that does NOT appear verbatim       #
    # ------------------------------------------------------------------ #
    keyword_hits = cm._search_contacts(filter="'preferences' in description")
    assert isinstance(keyword_hits, list) and len(keyword_hits) == 0

    # ------------------------------------------------------------------ #
    # 3️⃣  Nearest-neighbour search (k=1) – should pick Bob's entry      #
    # ------------------------------------------------------------------ #
    query = "favorite means of communication"
    nearest_k1 = cm._nearest_contacts(column="description", text=query, k=1)
    assert len(nearest_k1) == 1
    assert nearest_k1[0]["description"] == entries[1][1]  # Bob is best match

    # ------------------------------------------------------------------ #
    # 4️⃣  Nearest-neighbour search (k=2) – ordering + limit respected   #
    # ------------------------------------------------------------------ #
    nearest_k2 = cm._nearest_contacts(column="description", text=query, k=2)
    assert len(nearest_k2) == 2
    assert nearest_k2[0]["description"] == nearest_k1[0]["description"]
    remaining_descriptions = [
        e[1] for e in entries if e[1] != nearest_k1[0]["description"]
    ]
    assert nearest_k2[1]["description"] in remaining_descriptions

    # ------------------------------------------------------------------ #
    # 5️⃣  Derived vector column should now exist                         #
    # ------------------------------------------------------------------ #
    cols = cm._list_columns()
    assert "description_vec" in cols, "Vector column not created on-demand"
