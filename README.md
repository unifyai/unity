# unity

## Setup

### Create venv inside this folder
```
cd ~/unity (wherever you cloned it)
uv venv --python 3.11 .unity
source .unity/bin/activate
uv pip install -r requirements.txt

# Step 2: Install docling separately (avoids resolver deadlock)
uv pip install "docling>=2.49.0"
```

### Environment Variables

Populate an `.env` file in the same root directory (ie `~/unity/.env`), based on these newly generated keys, and also create some of your own for assistant customization (name, age etc.):
```
UNIFY_KEY={value}
UNIFY_BASE_URL={value}
# OFF_THE_SHELF=true # uncomment for using browser_use
OPENAI_API_KEY={value}
FIRST_NAME={value}
AGENT_FIRST={value}
AGENT_LAST={value}
AGENT_AGE={value}
DEEPGRAM_API_KEY={value}
CARTESIA_API_KEY={value}
LIVEKIT_URL={value}
LIVEKIT_API_KEY={value}
LIVEKIT_API_SECRET={value}
```

### Logging

Check out various logs in the "Assistants" project in the [Unity Interface](https://console.unify.ai/interfaces?project=Unity).

### Controller Mode

**Browser Mode**

1. Install the required dependencies through `node/npm`, then start the Magnitude server.

`npx ts-node agent-service/src/index.ts`

2. Use the actor in browser mode, i.e., `agent_mode="browser"`. This is the default mode.

**Desktop Mode**

1. Follow the guide in `desktop/README.md` for starting the virtual desktop and Magnitude server through Docker.

2. Use the actor in desktop mode, i.e., `agent_mode="desktop"`.
