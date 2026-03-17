# Demo Sites

Local website replicas for running end-to-end demos without real subscriptions.

Each subdirectory contains a self-contained web application that mimics a real
service (e.g. Zoho Connect). During demos, Playwright's `context.route()`
transparently reroutes requests from the real domain to the local replica, so
the browser (and both LLMs) believe they are interacting with the real site.

## Architecture

Demo sites are **started by agent-service** when it receives `urlMappings` in
the `/start` request. This ensures the demo site is always on the same machine
as the browser (magnitude), so `localhost` URLs work in all modes (web, web-vm,
desktop).

Demo sites use **pure HTML/CSS/JS** with a standalone `server.js` (Node.js
built-in `http` module, zero npm dependencies). No Python, no frameworks.

## Ports

Ports are assigned dynamically starting from 4001. Agent-service uses port 3000.

## Creating a New Demo Site

1. Create a directory: `demo-sites/<site-name>/`
2. Add a `server.js` that accepts a port as the first CLI argument:
   ```bash
   node server.js <port>
   ```
3. Reference the directory name in `unity/customization/clients/<client>/__init__.py`:

```python
register_org(
    org_id=...,
    config=ActorConfig(
        url_mappings={
            "https://www.example.com": "<site-name>",
        },
    ),
)
```

No registration in agent-service code is needed -- it discovers the directory
automatically at startup.

## How It Works

1. **Customization** (`unity/customization/`) -- per-org/team/user/assistant config
   defines `url_mappings` mapping real URLs to demo site directory names
   (e.g. `{"https://connect.zoho.com": "vantage-portal"}`)
2. **ComputerPrimitives** -- passes mappings to MagnitudeBackend
3. **Agent-service** `/start` -- receives `urlMappings`, calls `ensureDemoSites()`
   which finds the directory, assigns a free port, spawns `server.js`, and
   returns resolved `localhost` URLs
4. **BrowserConnector** -- registers `context.route()` handlers that intercept
   requests to the real domain and proxy them to the resolved `localhost:<port>`

The browser navigates to the real URL (e.g. `https://connect.zoho.com/`), but
Playwright intercepts the network request and serves it from the local demo
site. Since `page.url()` returns the original URL, all downstream consumers
(tab info, `get_content()`, `get_links()`, screenshots) see the real domain.

## Site Guidelines

- Sites should be pure HTML/CSS/JS with a standalone `server.js`
- Use Node.js built-in modules only (no npm dependencies)
- Accept the port as the first CLI argument
- Listen on `0.0.0.0` so the server is reachable from all interfaces
- Include realistic page structure and navigation to support demo workflows
- Avoid external API calls that would fail without real credentials
