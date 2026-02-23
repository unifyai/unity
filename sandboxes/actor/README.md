Actor Sandbox
=============

This folder contains an **interactive "flight simulator"** for the `CodeActActor` that lives in `unity/actor/`. The goal of this sandbox is to launch, monitor, and steer the actor as it works on a high-level goal.

[What are Actors?](https://github.com/unifyai/unity/blob/main/unity/actor/README.md)
----------------
**Actors** are the "brains" of the agent framework. They are responsible for taking a natural-language goal and executing a plan to achieve it.

* **`CodeActActor`** -- An agent that solves tasks primarily by writing and executing Python code in a REPL sandbox.

When an actor starts a task, it returns a `SteerableToolHandle`, which the sandbox uses to provide a rich, interactive control session.

## Prerequisites

### Magnitude Agent Service Setup
The `CodeActActor` requires the Magnitude agent service to be running for web automation tasks.

The repo uses Unity's modified `magnitude-core` for the agent service (see `agent-service/package.json` dependency: `"magnitude-core": "file:../magnitude/packages/magnitude-core"`). The `magnitude/` directory contains our fork with Unity-specific enhancements.

**1. Build local magnitude-core:**
```bash
cd magnitude && git checkout unity-modifications
cd packages/magnitude-core && npm install && npm run build
```

**2. Install Agent Service deps:**
```bash
cd agent-service && npm install
```

**3. Start the Service:**
```bash
cd agent-service && npx ts-node src/index.ts
```

The service will run on `http://localhost:3000` (configurable via `--agent-url`).

Running the sandbox
-------------------

```bash
# Run the CodeActActor in a text-only session
python -m sandboxes.actor.sandbox --actor code_act
```
