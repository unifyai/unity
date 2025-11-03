from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.web_searcher.web_searcher import WebSearcher


@pytest.mark.unit
@_handle_project
def test_websites_create_list_find_delete():
    ws = WebSearcher()

    out = ws._create_website(
        host="nytimes.com",
        gated=True,
        subscribed=False,
        credentials=None,
        actor_entrypoint=None,
        notes="Primary news source; check for general headlines.",
    )
    assert out["outcome"] == "website created"

    rows = ws._list_websites(filter="host == 'nytimes.com'")
    assert len(rows) == 1
    row = rows[0]
    assert row.host == "nytimes.com" and row.gated is True and row.subscribed is False
    assert isinstance(row.website_id, int) and row.website_id >= 0

    by_host = ws._find_websites(host="nytimes.com")
    assert len(by_host) == 1 and by_host[0].website_id == row.website_id

    by_id = ws._find_websites(website_id=row.website_id)
    assert len(by_id) == 1 and by_id[0].host == "nytimes.com"

    del_out = ws._delete_website(host="nytimes.com")
    assert del_out["outcome"] == "website deleted"
    assert ws._list_websites(filter="host == 'nytimes.com'") == []


@pytest.mark.unit
@_handle_project
def test_web_searcher_clear_resets_websites():
    ws = WebSearcher()
    ws._create_website(
        host="example.com",
        gated=False,
        subscribed=False,
        notes="Test site",
    )

    before = ws._list_websites()
    assert any(w.host == "example.com" for w in before)

    ws.clear()

    after = ws._list_websites()
    assert isinstance(after, list)
    assert all(w.host != "example.com" for w in after)
