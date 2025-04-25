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


### Debugging

To interact with the low-level browser controller, you can simply run `python test_controller.py` in the root directory, and it will spin up a GUI which shows the browser state and also the available actions (given the browser state). You can also send commands directly to the low-level LLM controller (the same one used in the voice demo above), to see how well the agent can convert english into playwright commands.

<div style="display: block;" align="center">
    <img class="dark-light" width="100%" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/refs/heads/main/img/externally_linked/unity_gui.png"/>
</div>
