# unity

## Setup

### Create venv inside this folder
```
cd ~/unity (wherever you cloned it)
uv venv .unity
source .unity/bin/activate
uv pip install -r requirements.txt
```

### Run LiveKit Demo

Run [this demo](https://docs.livekit.io/agents/start/voice-ai/#agent-code) (the SST-LLM-TTS one!), and create accounts for all third party platforms, storing the API Keys as needed.

### Environment Variables

Populate an `.env` file in the same root directory (ie `~/unity/.env`), based on these newly generated keys, and also create some of your own for assistant customization (name, age etc.):
```
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

### Download Required Files

With your venv activated (after calling `source .unity/bin/activate`):

```
python make_call.py download-files
```

### Have a Call!

With your venv activated:

```
python make_call.py
```

### Logging

Check out various logs in the "Unity" project in the [Unity Interface](https://console.unify.ai/interfaces?project=Unity).
