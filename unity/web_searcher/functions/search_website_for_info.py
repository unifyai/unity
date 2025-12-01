async def search_website_for_info(
    search_query,
    website,
    credentials,
    response_format=None,
):
    """
    Log into a website if needed and extract information relevant to a query.

    Behaviour
    ---------
    - Navigates to the provided website (prepends https:// when missing).
    - If credentials are provided as secret_ids, attempts to resolve a username/password
      pair via the SecretManager and log in when a sign-in flow is detected.
    - Searches the website for the given query, opens the best match and summarizes the result.
    - When a Pydantic response_format is provided, returns a structured object; otherwise returns a string summary.
    """

    from pydantic import BaseModel, Field
    from typing import Optional

    class ArticleExtract(BaseModel):
        """Structured extraction of article content."""

        title: str = Field(description="The article/page title")
        key_points: list[str] = Field(
            description="Key points relevant to the search query",
        )
        statistics: Optional[list[str]] = Field(
            default=None,
            description="Any important data or statistics mentioned",
        )

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
        f"Search for: '{search_query}', navigate to news, articles, insights, or research sections. "
        "Then, open the first article that appears in the search results.",
    )

    # Step 2: Review search results and identify relevant content
    # await action_provider.act(
    #     f"Review the search results or listings. Identify 2-3 articles or pages most relevant to '{search_query}'. "
    #     "Note their titles and URLs. Click on the most relevant one to open it. "
    #     "Use medium scrolls (at least 25px at a time) — be efficient.",
    # )

    # Step 3: Scroll through the first article to load content
    await action_provider.act(
        "Scroll through this article/page from top to bottom in large increments (full viewport at a time) to load all content. "
        "Do NOT use tiny scrolls — be efficient.",
    )

    # Step 4: Get the URL directly from browser state (reliable, no LLM)
    first_url = await action_provider.browser.get_current_url()

    # Step 5: Extract structured content using observe() with schema
    content_parts = []
    try:
        first_extract = await action_provider.observe(
            f"Extract the article content relevant to the query: '{search_query}'. "
            "Get the title, key points that answer the query, and any statistics or data mentioned.",
            response_format=ArticleExtract,
        )
        content_parts.append(
            f"**{first_extract.title}** ({first_url})\n"
            + "\n".join(f"- {p}" for p in first_extract.key_points),
        )
        if first_extract.statistics:
            content_parts[-1] += "\nStats: " + "; ".join(first_extract.statistics)
    except Exception:
        # Fallback to unstructured extraction if schema fails
        first_extract = await action_provider.observe(
            f"Summarize the key points of this article relevant to: '{search_query}'",
        )
        content_parts.append(f"({first_url})\n{first_extract}")

    # Step 6: Loop through additional articles (up to 2 more)
    visited_urls = {first_url}
    # for i in range(3):
    #     await action_provider.act(
    #         "Go back to the search results or listings. Open the next relevant article that hasn't been read yet. "
    #         f"Visited articles are: {visited_urls}. "
    #         "Use medium scrolls (at least 25px at a time) — be efficient.",
    #     )

    #     current_url = await action_provider.browser.get_current_url()
    #     if not current_url or current_url in visited_urls:
    #         continue  # No new article found, stop looping

    #     visited_urls.add(current_url)

    #     # Scroll through article to load content
    #     await action_provider.act(
    #         "Scroll through this article in large increments (full viewport at a time) to load all content. "
    #         "Do NOT use tiny scrolls — be efficient.",
    #     )

    #     try:
    #         extract = await action_provider.observe(
    #             f"Extract the article content relevant to the query: '{search_query}'. "
    #             "Get the title, key points, and any statistics.",
    #             response_format=ArticleExtract,
    #         )
    #         content_parts.append(
    #             f"**{extract.title}** ({current_url})\n"
    #             + "\n".join(f"- {p}" for p in extract.key_points),
    #         )
    #         if extract.statistics:
    #             content_parts[-1] += "\nStats: " + "; ".join(extract.statistics)
    #     except Exception:
    #         extract = await action_provider.observe(
    #             f"Summarize the key points of this article relevant to: '{search_query}'",
    #         )
    #         content_parts.append(f"({current_url})\n{extract}")

    # Step 8: Synthesize findings (direct LLM call - bypasses heavy context injection)
    import unify

    combined = "\n\n".join(content_parts)

    summarize_client = unify.AsyncUnify("gpt-5@openai")
    summarize_client.set_system_message(
        "You are a concise summarization assistant. Synthesize content into clear, factual summaries with citations.",
    )
    summary = await summarize_client.generate(
        f"Synthesize the following extracted content into a concise summary (3-5 sentences) answering the query: '{search_query}'. "
        f"The source URLs are already included — preserve them as citations.\n\n{combined}",
    )

    if response_format is not None:
        try:
            from pydantic import BaseModel

            if isinstance(response_format, type) and issubclass(
                response_format,
                BaseModel,
            ):
                struct_client = unify.AsyncUnify("gpt-5@openai")
                struct_client.set_system_message(
                    "Transform the given content into the requested structured format. Only include fields that exist on the model.",
                )
                struct_client.set_response_format(response_format)
                raw = await struct_client.generate(
                    f"Content to transform:\n\n{summary}",
                )
                return response_format.model_validate_json(raw)
        except Exception:
            return str(summary)

    return str(summary)
