# Linux Remote Client

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

### Troubleshooting

- Make sure `ANTHROPIC_API_KEY`, `UNIFY_BASE_URL` and `UNIFY_KEY` are in your `.env` file when starting the Docker container.
- When running with Actor, make sure `UNIFY_KEY` and at least `ASSISTANT_EMAIL=unity.agent@unity.ai` are present in your unity `.env` for the magnitude server auth to work.
