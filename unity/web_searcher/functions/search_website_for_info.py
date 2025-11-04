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

    await action_provider.act(
        (
            "Use the site's own search or navigation to find the single most relevant page for: '%s'. Open it, read it, and prepare a short summary with the page URL."
            % (search_query,)
        ),
    )
    summary = await action_provider.query(
        "Return a concise summary (3-5 sentences) and include the current page URL at the top.",
    )

    if response_format is not None:
        try:
            structured = await action_provider.reason(
                request=(
                    "Transform the given page summary into the requested structured model. Only include fields that exist on the model."
                ),
                context=str(summary),
                response_format=response_format,
            )
            return structured
        except Exception:
            return str(summary)

    return str(summary)
