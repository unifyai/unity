async def search_website_for_info(
    search_query,
    website,
    credentials,
):
    """
    Log into a website if needed and extract information relevant to a query.

    Behaviour
    ---------
    - Navigates to the provided website (prepends https:// when missing).
    - If credentials are provided as secret_ids, attempts to resolve a username/password
      pair via the SecretManager and log in when a sign-in flow is detected.
    - Searches the website for the given query, opens the best match and summarizes the result.
    - Returns a string summary.
    """

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

    # Step 1: Find and use the site's search functionality
    await action_provider.act(
        f"Look for a search box or search icon on this page. If you find one, use it to search for: '{search_query}'. "
        "Use plain words only — do not use special syntax, operators, or quotes. Modify query if no results found. "
        "Only if no search functionality exists, navigate to a news, articles, insights, or research section instead.",
    )

    # Step 2: Get all links from the search results page
    print("[WS] Extracting links from search results...")
    links_result = await action_provider.get_links(same_domain=True)
    links = links_result.get("links", [])
    print(f"[WS] Found {len(links)} links")

    # Step 3: Extract content from top N articles using raw content extraction
    n = 5
    content_parts = []
    visited_urls = set()

    # Skip common non-content URL patterns
    skip_patterns = [
        "login",
        "signin",
        "signup",
        "register",
        "search?",
        "newsletter",
        "account",
        "subscribe",
        "/category/",
        "/author/",
        "/page/",
        "#",
        "javascript:",
        "mailto:",
    ]

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
            url = content_result.get("url", href)
            title = content_result.get("title", "") or link_text
            raw_content = content_result.get("content", "")

            # Truncate content for LLM context window
            truncated = raw_content[:4000] if len(raw_content) > 4000 else raw_content

            if truncated.strip():
                content_parts.append(f"**{title}** ({url})\n{truncated}")
                print(f"[WS] Extracted content from: {title[:50]}...")

        except Exception as e:
            print(f"[WS] Failed to extract from {href[:50]}: {e}")
            continue

    return "\n\n".join(content_parts)
