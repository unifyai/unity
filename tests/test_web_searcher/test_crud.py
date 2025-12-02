from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unity.web_searcher.web_searcher import WebSearcher


@_handle_project
def test_websites_create_filter_search_delete():
    ws = WebSearcher()

    out = ws._create_website(
        name="The New York Times",
        host="nytimes.com",
        gated=True,
        subscribed=False,
        credentials=None,
        actor_entrypoint=None,
        notes="Primary news source; check for general headlines.",
    )
    assert out["outcome"] == "website created"

    rows = ws._filter_websites(filter="host == 'nytimes.com'")
    assert len(rows) == 1
    row = rows[0]
    assert row.name == "The New York Times"
    assert row.host == "nytimes.com" and row.gated is True and row.subscribed is False
    assert isinstance(row.website_id, int) and row.website_id >= 0

    by_host = ws._filter_websites(filter="host == 'nytimes.com'")
    assert len(by_host) == 1 and by_host[0].website_id == row.website_id

    by_id = ws._filter_websites(filter=f"website_id == {row.website_id}")
    assert len(by_id) == 1 and by_id[0].host == "nytimes.com"

    del_out = ws._delete_website(host="nytimes.com")
    assert del_out["outcome"] == "website deleted"
    assert ws._filter_websites(filter="host == 'nytimes.com'") == []


@_handle_project
def test_clear_resets_websites():
    ws = WebSearcher()
    ws._create_website(
        name="Example",
        host="example.com",
        gated=False,
        subscribed=False,
        notes="Test site",
    )

    before = ws._filter_websites()
    assert any(w.host == "example.com" for w in before)

    ws.clear()

    after = ws._filter_websites()
    assert isinstance(after, list)
    assert all(w.host != "example.com" for w in after)


@_handle_project
def test_update_website_by_id_host_and_name():
    ws = WebSearcher()

    # Create two distinct websites
    ws._create_website(
        name="NYTimes",
        host="nytimes.com",
        gated=True,
        subscribed=False,
        notes="Primary news",
    )
    ws._create_website(
        name="ArXiv",
        host="arxiv.org",
        gated=False,
        subscribed=False,
        notes="Preprints",
    )

    # Fetch id for NYTimes
    rows = ws._filter_websites(filter="host == 'nytimes.com'", limit=1)
    assert rows and rows[0].name == "NYTimes"
    ny_id = rows[0].website_id

    # Update by id (change name and toggle subscribed)
    out = ws._update_website(
        website_id=ny_id,
        name="The New York Times",
        subscribed=True,
    )
    assert out["outcome"] == "website updated"
    after = ws._filter_websites(filter=f"website_id == {ny_id}", limit=1)[0]
    assert after.name == "The New York Times" and after.subscribed is True

    # Update by host (change host)
    out2 = ws._update_website(
        match_host="nytimes.com",
        host="nytimes.com/news",
    )
    assert out2["outcome"] == "website updated"
    # Should now be discoverable by the new host
    moved = ws._filter_websites(filter="host == 'nytimes.com/news'", limit=1)
    assert moved and moved[0].website_id == ny_id

    # Update by name (change gated)
    out3 = ws._update_website(
        match_name="The New York Times",
        gated=False,
    )
    assert out3["outcome"] == "website updated"
    final_row = ws._filter_websites(filter="name == 'The New York Times'", limit=1)[0]
    assert final_row.gated is False


@_handle_project
def test_update_website_enforces_uniqueness():
    ws = WebSearcher()

    ws._create_website(
        name="GitHub",
        host="github.com",
        gated=True,
        subscribed=True,
        notes="Dev hosting",
    )
    ws._create_website(
        name="GitLab",
        host="gitlab.com",
        gated=False,
        subscribed=False,
        notes="Dev hosting 2",
    )

    # Attempt to update GitLab host to an existing host (github.com)
    with pytest.raises(AssertionError):
        ws._update_website(match_name="GitLab", host="github.com")

    # Attempt to update GitLab name to an existing name (GitHub)
    with pytest.raises(AssertionError):
        ws._update_website(match_host="gitlab.com", name="GitHub")


@_handle_project
def test_update_website_edge_cases():
    ws = WebSearcher()

    ws._create_website(
        name="Medium",
        host="medium.com",
        gated=True,
        subscribed=True,
        notes="Writing platform",
    )

    # No identifier provided
    with pytest.raises(ValueError):
        ws._update_website()

    # Identifier provided but no updates
    with pytest.raises(ValueError):
        ws._update_website(match_host="medium.com")

    # Not found
    with pytest.raises(ValueError):
        ws._update_website(match_name="NonExistent", notes="none")
