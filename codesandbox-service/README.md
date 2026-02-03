# CodeSandbox SDK Service

This Node.js service acts as an HTTP wrapper for the CodeSandbox SDK, exposing filesystem and convenience file APIs so our Python clients can create/connect sandboxes and read/write/rename/remove files programmatically.

## Setup

1.  Install Dependencies: Ensure you have Node.js installed.

2.  Install service deps:

    ```bash
    cd codesandbox-service
    npm install
    ```

3.  Create Environment File: This service requires a CodeSandbox API token and Unify settings for request authentication.

    Create a `.env` file in this directory:

    ```bash
    # codesandbox-service/.env
    CODESANDBOX_API_TOKEN="csb-..."     # CodeSandbox SDK token
    CODESANDBOX_TEMPLATE_ID="..."       # Optional: template to seed new sandboxes
    CODESANDBOX_SERVICE_PORT="3100"     # Optional: service port (defaults to 3100)

    # Auth verification against Unify
    ORCHESTRA_URL="..."
    UNIFY_KEY="..."
    ```

    Requests must include an Authorization header in the format:

    ```
    Authorization: Bearer <UNIFY_KEY> <ASSISTANT_EMAIL>
    ```

## Running the Service

You can run the service by:

```bash
npx ts-node src/index.ts
```

The service will start and listen on `http://localhost:3100` (or `CSB_SERVICE_PORT`).

## API Endpoints

-   `POST /sandboxes/create`: Creates a new sandbox (uses optional `CODESANDBOX_TEMPLATE_ID`).
-   `POST /sandboxes/:id/connect`: Opens a sandbox handle and caches it for subsequent FS ops.

Filesystem endpoints (operate on an opened sandbox):

-   `GET /fs/:id/readdir?dir=/project` — list directory entries
-   `GET /fs/:id/readFile?path=/project/file.txt` — read file bytes
-   `POST /fs/:id/writeFile` — body: `{ path, data, encoding? }`
-   `POST /fs/:id/rename` — body: `{ oldPath, newPath }` (uses `mv` under the hood)
-   `POST /fs/:id/move` — body: `{ oldPath, newParentPath }` (uses `mv`)
-   `POST /fs/:id/mkdir` — body: `{ path }`
-   `POST /fs/:id/remove` — body: `{ path, recursive }` (`rm -rf` when `recursive=true`)
-   `GET /fs/:id/stat?path=/project` — best-effort stat

High-level file endpoints (project/filename semantics, mirrors our Next.js reference):

-   `POST /file` — `{ user_id, project, filename, content }`
    - Special case: when `filename === ".env"` and `content === ""`, service writes a minimal `.env` with `UNIFY_KEY` and `UNIFY_PROJECT`.
-   `DELETE /file` — `{ user_id, project, filename, isDirectory? }`
-   `PUT /file` — `{ user_id, project, old_filename, new_filename }`
-   `GET /file?user_id=...&project=...&filename=...&isDirectory=true|false`

-   `GET /health`: Checks if the service is ready to accept requests.

## Notes

- Authentication is verified against `ORCHESTRA_URL` using the `Authorization: Bearer <UNIFY_KEY> <ASSISTANT_EMAIL>` header, mirroring our agent-service scheme.
- For recursive delete and move/rename, the service uses `sandbox.shells.run("rm -rf ..." | "mv ...")` to match the proven reference behavior.
- If you see type/module resolution errors locally, ensure `npm install` has run; the Docker build also runs `npm ci` for this service.
