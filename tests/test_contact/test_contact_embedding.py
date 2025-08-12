"""
Vector-embedding & semantic-search tests for **ContactManager**.

We mirror the pattern used in `test_knowledge_embedding` but operate on the
single contacts table, exercising `_search_contacts` and the automatic creation
of `<source>_vec` derived columns.
"""

import pytest

from unity.contact_manager.contact_manager import ContactManager

# keeps each test isolated in its own Unify project / trace context
from tests.helpers import _handle_project


@pytest.mark.unit
@pytest.mark.requires_real_unify
@_handle_project
def test_contact_embedding_and_search():
    cm = ContactManager()

    # ------------------------------------------------------------------ #
    # 1️⃣  Create three contacts whose *bio* fields are related  #
    # ------------------------------------------------------------------ #
    entries = [
        ("Alice", "I email and phone sometimes."),
        ("Bob", "Text messaging is my go-to communication method."),
        ("Carol", "I love taking the train to work."),
    ]
    for fname, desc in entries:
        cm._create_contact(first_name=fname, bio=desc)

    # ------------------------------------------------------------------ #
    # 2️⃣  Keyword search for a term that does NOT appear verbatim       #
    # ------------------------------------------------------------------ #
    keyword_hits = cm._filter_contacts(filter="'preferences' in bio")
    assert isinstance(keyword_hits, list) and len(keyword_hits) == 0

    # ------------------------------------------------------------------ #
    # 3️⃣  Nearest-neighbour search (k=1) – should pick Bob's entry      #
    # ------------------------------------------------------------------ #
    query = "favorite means of communication"
    nearest_k1 = cm._search_contacts(references={"bio": query}, k=1)
    assert len(nearest_k1) == 1
    assert nearest_k1[0].bio == entries[1][1]  # Bob is best match

    # ------------------------------------------------------------------ #
    # 4️⃣  Nearest-neighbour search (k=2) – ordering + limit respected   #
    # ------------------------------------------------------------------ #
    nearest_k2 = cm._search_contacts(references={"bio": query}, k=2)
    assert len(nearest_k2) == 2
    assert nearest_k2[0].bio == nearest_k1[0].bio
    remaining_descriptions = [e[1] for e in entries if e[1] != nearest_k1[0].bio]
    assert nearest_k2[1].bio in remaining_descriptions

    # ------------------------------------------------------------------ #
    # 5️⃣  Derived vector column should now exist                         #
    # ------------------------------------------------------------------ #
    cols = cm._list_columns()
    assert "_bio_emb" in cols, "Vector column not created on-demand"
