# MacOS Remote Client

### Prerequisites

1. Homebrew

### Setup

1. Install the required package

`bash install.sh`

2. Start the remote client app.

`bash remote.sh $UNIFY_KEY`

3. Tunnel the service to HTTPS.

a. For testing

- Start the tunnel. A URL for testing will be provided.

`bash tunnel.sh`

b. For production - WIP

- Login to Cloudflare. This is a one time step.

`cloudflared tunnel login`

- Start the tunnel.

`TUNNEL_HOSTNAME=<prod_hostname> TUNNEL_NAME=<prod_appname> bash tunnel.sh`

### Live Remote Viewing and Controls

1. Tunnel the remote view.

`bash liveview.sh`

2. View and control the desktop through the URL below. When prompted for password, input your Unify API key.

`<cloudflared-url>/vnc.html?resize=scale&autoreconnect=1&autoconnect=1`

### Troubleshooting

- Make sure `ANTHROPIC_API_KEY`, `UNIFY_BASE_URL` and `UNIFY_KEY` are in your `.env` file when starting the Docker container.
- When running with Actor, make sure `UNIFY_KEY` and at least `ASSISTANT_EMAIL=unity.agent@unity.ai` are present in your unity `.env` for the magnitude server auth to work.
