# Virtual Desktop Runtime

This directory provides a Node-based runtime that launches:
- A lightweight Linux desktop (Xvfb + Fluxbox + x11vnc + noVNC)
- Virtual audio devices (PipeWire/WirePlumber + PulseAudio modules)
- The Magnitude BrowserAgent (`agent-service`)

## What’s included
- Dockerfile: Based on `node:20-bookworm-slim` with only the required deps
  - Virtual desktop/remote view (Xvfb, Fluxbox, x11vnc, websockify, noVNC assets)
  - Browser runtime libraries (Chromium/Playwright)
  - Virtual audio stack (pipewire, wireplumber, pulseaudio/alsa utils)
  - Native module toolchain/runtime (`build-essential`, `python3`, `pkg-config`, `libvips`)
- desktop.sh: Starts Xvfb, window manager, x11vnc, noVNC, xdg-desktop-portal; creates `/tmp/unify/assistant/install`
- device.sh: Initializes PipeWire/WirePlumber and null-sink routing via PulseAudio
- startup.sh: Orchestrates startup and clean shutdown (traps SIGTERM/SIGINT)

### Exposed ports
- 6080/tcp: noVNC web UI
- 5900/tcp: x11vnc server
- 3000/tcp: agent-service HTTP API

### Shutdown behavior
startup.sh records PIDs of desktop, device and agent processes and on SIGTERM/SIGINT sends TERM and waits for each to exit, ensuring a clean shutdown.

## Setup
- Build: `docker build -t unity-desktop -f desktop/Dockerfile .`
- Run: `docker run --rm -p 6080:6080 -p 5900:5900 -p 3000:3000 --env-file .env unity-desktop`
- Open `http://localhost:6080/vnc.html?resize=scale&autoconnect=1&autoreconnect=1` to view the desktop
- Control: You can now run the Actor/Controller locally with `agent_mode="desktop"` (e.g., `HierarchicalActor(agent_mode="desktop")` or `Browser(mode="magnitude", agent_mode="desktop")`)

### Troubleshooting
- Make sure `ANTHROPIC_API_KEY`, `UNIFY_BASE_URL` and `UNIFY_KEY` are in your `.env` file when starting the Docker container.
- When running with Actor, make sure `UNIFY_KEY` and at least `ASSISTANT_EMAIL=unity.agent@unity.ai` are present in your unity `.env` for the magnitude server auth to work.
