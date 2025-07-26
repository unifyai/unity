# unity

## Setup

### Create venv inside this folder
```
cd ~/unity (wherever you cloned it)
uv venv --python 3.11 .unity
source .unity/bin/activate
uv pip install -r requirements.txt
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
