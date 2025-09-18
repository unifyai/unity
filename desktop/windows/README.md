# Windows Remote Client

### Prerequisites

1. PowerShell

2. Chocolatey

```powershell
winget install --id=Chocolatey.Chocolatey -e
```

3. Git

```powershell
winget install --id=Git.Git -e --source winget
```

4. Python 3 (for websockify)

```powershell
winget install --id=Python.Python.3 -e
```

### Setup

Watch this video for [local setup](https://www.loom.com/share/61a230c7d7314a109e3fc64061d8e315?sid=b80b1f19-c080-4431-a667-6ee1a0c350f1).

1. Install the required package through PowerShell in "Run as Administrator" mode.

```powershell
.\install.ps1
```

- When prompted by a TightVNC popup window, set/change primary password to your Unify API key.

2. Start the remote client app.

```powershell
.\remote.ps1
```

3. Tunnel the service to HTTPS.

a. For testing

- Start the tunnel. A URL for testing will be provided.

```powershell
.\tunnel.ps1
```

b. For production - WIP

- Login to Cloudflare. This is a one time step.

```powershell
cloudflared tunnel login
```

- Start the tunnel - TODO

```powershell
.\tunnel.ps1 -Hostname your.domain.com -TunnelName myapp -LocalPort 6080
```

### Live Remote Viewing and Controls

1. Tunnel the remote view.

```powershell
.\liveview.ps1
```

2. View and control the desktop through the URL below. When prompted for password, input your Unify API key.

`<cloudflared-url>/vnc.html?resize=scale&autoreconnect=1&autoconnect=1`

### Troubleshooting

- Make sure `ANTHROPIC_API_KEY`, `UNIFY_BASE_URL` and `UNIFY_KEY` are in your `.env` file when starting the Docker container.
- When running with Actor, make sure `UNIFY_KEY` and at least `ASSISTANT_EMAIL=unity.agent@unity.ai` are present in your unity `.env` for the magnitude server auth to work.
