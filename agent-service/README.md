# Magnitude BrowserAgent Service

This Node.js service acts as an HTTP wrapper for the Magnitude `BrowserAgent`, allowing a Python client (like the Hierarchical Actor) to perform autonomous web automation tasks.

## Setup

1.  **Install Dependencies**: Ensure you have Node.js installed. Then, from the `agent-service` directory, run:
    ```bash
    npm install
    ```

2.  **Create Environment File**: This service requires an API key for the underlying Large Language Model. Create a `.env` file in the root of this directory:
    ```
    # agent-service/.env
    ANTHROPIC_API_KEY="sk-ant-..."
    UNIFY_BASE_URL="..."
    UNIFY_KEY="..."
    ```

## Running the Service

You can run the service by:

```bash
npx ts-node src/index.ts
```

The service will start and listen on `http://localhost:3000`.

## API Endpoints

-   `POST /nav`: Navigates the browser to a URL.
-   `POST /act`: Executes a high-level task on the current page.
-   `POST /extract`: Extracts structured data from the current page.
-   `GET /screenshot`: Returns a base64-encoded screenshot of the current page.
-   `POST /stop`: Gracefully shuts down the agent and browser.
-   `GET /health`: Checks if the service is ready to accept requests.
