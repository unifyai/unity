# Demo Sites

Local website replicas for running end-to-end demos without real subscriptions.

Each subdirectory contains a self-contained web application that mimics a real
service (e.g. Zoho CRM, Salesforce). During demos, Playwright's `context.route()`
transparently reroutes requests from the real domain to the local replica, so
the browser (and both LLMs) believe they are interacting with the real site.

## Quick Start

```bash
docker compose up -d
```

This starts all demo sites. Each site is mapped to a unique host port (4001+).

## Port Convention

| Port  | Site                |
|-------|---------------------|
| 4001  | example (Pawsome Dog Rescue) |
| 4002+ | (future demo sites) |

Agent-service uses port 3000. Demo sites start at 4001 to avoid conflicts.

## Creating a New Demo Site

1. Create a directory: `demo-sites/<site-name>/`
2. Add a `Dockerfile` that serves the site on port 3000 internally
3. Add the service to `docker-compose.yml` with the next available host port
4. Register the URL mapping in `unity/customization/clients/<client>/__init__.py`:

```python
register_org(
    org_id=...,
    config=ActorConfig(
        url_mappings={
            "https://www.example.com": "http://localhost:<port>",
        },
    ),
)
```

## How It Works

The URL mapping flows through:

1. **Customization** (`unity/customization/`) -- per-org/team/user/assistant config
2. **ComputerPrimitives** -- passes mappings to MagnitudeBackend
3. **Agent-service** `/start` -- forwards `urlMappings` to magnitude
4. **BrowserConnector** -- registers `context.route()` handlers on the Playwright context

The browser navigates to the real URL (e.g. `https://www.zoho.com/`), but
Playwright intercepts the network request and serves it from localhost. Since
`page.url()` returns the original URL, all downstream consumers (tab info,
`get_content()`, `get_links()`, screenshots) see the real domain.

## Site Guidelines

- Sites should be static or lightweight (Next.js, plain HTML, etc.)
- Internal port should be 3000 (standard for Next.js / static servers)
- Include realistic page structure and navigation to support demo workflows
- Avoid external API calls that would fail without real credentials
