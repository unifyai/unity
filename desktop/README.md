# Virtual Desktop Runtime

This directory provides a Node-based runtime that launches:
- A lightweight Linux desktop (Xvfb + Fluxbox + x11vnc + noVNC)
- Virtual audio devices (PipeWire/WirePlumber + PulseAudio modules)
- The Magnitude BrowserAgent (`agent-service` and `magnitude`)

## CLI Tools

Installations are available in the `unity-desktop-assistant` repo.

1. [Ubuntu](https://github.com/unifyai/unify-desktop-assistant/tree/ubuntu)
2. [Windows](https://github.com/unifyai/unify-desktop-assistant/tree/win)
3. [MacOS](https://github.com/unifyai/unify-desktop-assistant/tree/macos)

## Setup

Watch these videos for [setup](https://www.loom.com/share/ad1a9b9c4e6e4de1a9b5012523a34049?sid=974205eb-28ad-4f91-9752-dfe722fed214) and [controls](https://www.loom.com/share/654bcf321cd24a698032dab5e7e45059?sid=ea521290-cf86-425c-8308-6845864c86ee).

- Build: `docker build -t unity-desktop -f desktop/Dockerfile .`
- Run: `docker run --rm -p 6080:6080 -p 5900:5900 -p 3000:3000 --env-file .env unity-desktop`
- Open `http://localhost:6080/custom.html?password=<UNIFY_KEY>` to view the desktop (clean UI, no sidebar/logo)
- Control: Use `primitives.computer.desktop.*` for full desktop control (mouse/keyboard). Use `primitives.computer.web.new_session(visible=True/False)` to create browser sessions (visible on VM or headless).

### Troubleshooting
- Make sure `UNITY_COMMS_URL`, `ORCHESTRA_URL` and `UNIFY_KEY` are in your `.env` file when starting the Docker container.
- Also make sure `UNITY_COMMS_URL`, `ORCHESTRA_URL` and `UNIFY_KEY` are in your `agent-service/.env` file on starting the server.
- When running with Actor, make sure `UNIFY_KEY` is present in your unity `.env` for the agent-service server auth to work.
