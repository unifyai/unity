async def search_website_for_info(
    search_queries,
    website,
    credentials,
):
    """
    Log into a website if needed and extract information relevant to one or more queries.

    Behaviour
    ---------
    - Navigates to the provided website (prepends https:// when missing).
    - If credentials are provided as secret_ids, attempts to resolve a username/password
      pair via the SecretManager and log in when a sign-in flow is detected.
    - Searches the website for each query in search_queries, extracts content from results.
    - Returns a string summary with all extracted content.

    Parameters
    ----------
    search_queries : list[str] | str
        One or more search queries to run on the site. Can be a single string or list.
    website : str
        The website host/URL to search.
    credentials : list[int]
        List of secret_ids for login credentials.
    """
    # Normalize search_queries to a list
    if isinstance(search_queries, str):
        search_queries = [search_queries]

    url = (website or "").strip()
    if url and not (url.startswith("http://") or url.startswith("https://")):
        url = f"https://{url}"
    await action_provider.navigate(url)

    try:
        sm = action_provider.secret_manager
        names = []
        for sid in credentials or []:
            try:
                rows = sm._filter_secrets(filter=f"secret_id == {int(sid)}", limit=1)
                if rows:
                    nm = rows[0].name
                    if isinstance(nm, str) and nm:
                        names.append(nm)
            except Exception:
                continue
    except Exception:
        pass

    # Provide explicit placeholder hints to the actor instead of resolving values here
    if names:
        hints = []
        for nm in names:
            hints.append(f"use ${{{nm}}} for {nm}")
        hints_text = "; ".join(hints)
        hints_text = await action_provider.secret_manager.from_placeholder(hints_text)
        await action_provider.act(
            (
                "If a login is required for this site, complete the sign-in form using the available secret placeholders; "
                f"{hints_text}. Do not echo raw values. After logging in, confirm you are authenticated and can access gated pages."
            ),
        )
    else:
        await action_provider.act(
            "If a login screen appears and credentials are required, indicate that no usable credentials were resolved and continue with public content.",
        )

    # Skip common non-content URL patterns
    skip_patterns = [
        "login",
        "logout",
        "signin",
        "signup",
        "signout",
        "register",
        "search?",
        "newsletter",
        "account",
        "subscribe",
        "users",
        "edit",
        "/category/",
        "/author/",
        "/page/",
        "#",
        "javascript:",
        "mailto:",
        "filter",
        "tracker",
    ]

    all_content_parts = []
    visited_urls = set()

    # Loop through each search query
    for search_query in search_queries:
        print(f"[WS] Searching for: {search_query}")

        # Navigate back to homepage for each new query
        # await action_provider.navigate(url)

        # Step 1: Find and use the site's search functionality
        await action_provider.act(
            f"Look for a search box or search icon. Try home page if not found. If you find one, use it to search for: '{search_query}'. "
            "Use plain words only for general searches — do not use special syntax, operators, or quotes. "
            "If location search is involed, use town names, and filter distance within reasonable range if applicable. "
            "If you see '0 results', 'no results found', or an error message, try a simpler/shorter query. "
            "Only if no search functionality exists, navigate to a query-relevant section (e.g., news, articles, deals, etc) instead.",
        )

        # Step 2: Get all links from the search results page
        print("[WS] Extracting links from search results...")
        links_result = await action_provider.get_links(same_domain=True)
        links = links_result.get("links", [])
        print(f"[WS] Found {len(links)} links for query: {search_query}")

        # Step 3: Extract content from top N articles using raw content extraction
        n = 3
        i = 0
        for link in links[:100]:
            if i >= n:
                break

            href = link.get("href", "")
            link_text = link.get("text", "")

            # Skip empty, visited, or obviously non-article links
            if not href or href in visited_urls:
                continue
            if any(skip in href.lower() for skip in skip_patterns):
                continue
            if href.endswith("/"):
                continue

            visited_urls.add(href)
            i += 1

            try:
                print(f"[WS] Navigating to: {href[:80]}...")
                await action_provider.navigate(href)

                # Check for and solve any captcha if present
                await action_provider.act(
                    "If you see a CAPTCHA, cookie consent, or access verification prompt, solve or dismiss it. "
                    "Otherwise, do nothing and confirm the page content is visible.",
                )

                # Get raw content in markdown format (no LLM overhead)
                content_result = await action_provider.get_content(format="markdown")
                page_url = content_result.get("url", href)
                title = content_result.get("title", "") or link_text
                raw_content = content_result.get("content", "")

                # Truncate content for LLM context window
                truncated = (
                    raw_content[:4000] if len(raw_content) > 4000 else raw_content
                )

                if truncated.strip():
                    all_content_parts.append(
                        f"**{title}** ({page_url})\n[Query: {search_query}]\n{truncated}",
                    )
                    print(f"[WS] Extracted content from: {title[:50]}...")

            except Exception as e:
                print(f"[WS] Failed to extract from {href[:50]}: {e}")
                continue

    return "\n\n".join(all_content_parts)
