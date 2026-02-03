# Virtual Desktop Runtime

This directory provides a Node-based runtime that launches:
- A lightweight Linux desktop (Xvfb + Fluxbox + x11vnc + noVNC)
- Virtual audio devices (PipeWire/WirePlumber + PulseAudio modules)
- The Magnitude BrowserAgent (`agent-service`)

## CLI Tools

Installations are available in the `unity-desktop-assistant` repo.

1. [Ubuntu](https://github.com/unifyai/unify-desktop-assistant/tree/ubuntu)
2. [Windows](https://github.com/unifyai/unify-desktop-assistant/tree/win)
3. [MacOS](https://github.com/unifyai/unify-desktop-assistant/tree/macos)

## Setup

Watch these videos for [setup](https://www.loom.com/share/ad1a9b9c4e6e4de1a9b5012523a34049?sid=974205eb-28ad-4f91-9752-dfe722fed214) and [controls](https://www.loom.com/share/654bcf321cd24a698032dab5e7e45059?sid=ea521290-cf86-425c-8308-6845864c86ee).

- Build: `docker build -t unity-desktop -f desktop/Dockerfile .`
- Run: `docker run --rm -p 6080:6080 -p 5900:5900 -p 3000:3000 --env-file .env unity-desktop`
- Open `http://localhost:6080/custom.html` to view the desktop (clean UI, no sidebar/logo)
- Control: You can now run the Actor/Controller locally with `agent_mode="desktop"` (e.g., `HierarchicalActor(agent_mode="desktop")` or `Browser(mode="magnitude", agent_mode="desktop")`)

### Troubleshooting
- Make sure `ANTHROPIC_API_KEY`, `ORCHESTRA_URL` and `UNIFY_KEY` are in your `.env` file when starting the Docker container.
- When running with Actor, make sure `UNIFY_KEY` and at least `ASSISTANT_EMAIL=unity.agent@unity.ai` are present in your unity `.env` for the magnitude server auth to work.
