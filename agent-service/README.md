# Magnitude BrowserAgent Service

This Node.js service acts as an HTTP wrapper for the Magnitude `BrowserAgent`, allowing a Python client (like the Hierarchical Actor) to perform autonomous web automation tasks.

## Setup

1.  **Install Dependencies**: Ensure you have Node.js installed.

2.  **Build local magnitude-core (Unity fork setup)**:

    This repo uses a local checkout of Unity's modified `magnitude-core` via a file dependency (see `package.json`: `"magnitude-core": "file:../magnitude/packages/magnitude-core"`). The `magnitude/` directory contains our fork of the magnitude repository with Unity-specific enhancements.

    ```bash
    # First, clone the unity repository if you haven't already
    git clone <unity-repo-url>
    cd unity
    
    # Clone Unity's magnitude fork into the magnitude/ subdirectory
    git clone https://github.com/unifyai/magnitude.git magnitude
    cd magnitude
    git checkout unity-modifications  # Our branch with Unity enhancements
    
    # Build magnitude-core
    cd packages/magnitude-core
    npm install
    npm run build
    # optional: run tests/examples if needed
    ```

3.  **Install service deps**:

    From the `agent-service` directory, run:
    ```bash
    cd ../../..
    cd agent-service
    npm install
    ```

4.  **Create Environment File**: This service requires an API key for the underlying Large Language Model and Unify.
    Create a `.env` file in the root of this directory:
    ```
    # agent-service/.env
    ANTHROPIC_API_KEY="sk-ant-..."    # or provide other LLM keys used by your magnitude-core config
    UNIFY_BASE_URL="..."
    UNIFY_KEY="..."
    # Optional keys depending on configured clients in magnitude-core (baml clients)
    GOOGLE_API_KEY="..."              # if using Google AI Studio clients
    OPENROUTER_API_KEY="..."          # if using OpenRouter
    OPENAI_API_KEY="..."              # if using OpenAI
    ```

## Running the Service

You can run the service by:

```bash
npx ts-node src/index.ts
```

The service will start and listen on `http://localhost:3000`.

## Developing with local magnitude-core changes

If you modify code in `magnitude/packages/magnitude-core`, rebuild it and refresh the local dependency in this service:

```bash
# In magnitude-core
cd magnitude/packages/magnitude-core
npm run build

# Back in agent-service - reinstall to pick up the updated local package
cd ../..
cd agent-service
npm install --force   # ensures the local file: dependency is re-copied
```

### Working with the Unity Fork

The `magnitude/` directory is our fork of the magnitude repository with Unity-specific modifications. Key points:

- **Branch**: Always work on `unity-modifications` branch
- **Upstream sync**: Use `upstream-main` branch to pull in latest magnitude changes
- **Team sharing**: Push your changes to `https://github.com/unifyai/magnitude.git`

See `MAGNITUDE_SETUP.md` in the repo root for detailed workflow instructions.

Notes:
- If you see runtime errors such as "Cannot find module './dist/...'", it means `magnitude-core` has not been built. Run `npm run build` in `magnitude-core`.
- For a tighter inner loop, you can also use `yalc` (optional):
  - In `magnitude-core`: `npm run build && npx yalc publish --push`
  - In `agent-service`: `npx yalc add magnitude-core`
  - Re-run the publish step after changes to auto-push updates.

## API Endpoints

-   `POST /nav`: Navigates the browser to a URL.
-   `POST /act`: Executes a high-level task on the current page.
-   `POST /extract`: Extracts structured data from the current page.
-   `GET /screenshot`: Returns a base64-encoded screenshot of the current page.
-   `POST /stop`: Gracefully shuts down the agent and browser.
-   `GET /health`: Checks if the service is ready to accept requests.
