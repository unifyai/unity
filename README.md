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

### Desktop Control

1. Build the image.

`docker build --build-arg UNIFY_KEY=<your-key> -t unity_browser:latest -f Dockerfile .`

2. Start the container.

`docker run -it -p 8080:8080 -p 6080:6080 --env-file .env -v $(pwd):/app --rm unity_browser:latest bash`

3. In the container, start the virtual desktop.

`bash desktop.sh`

4. Access the virtual desktop through the address below. It can now be used with the planner and the Magnitude controller.

`http://localhost:6080/vnc.html?resize=scale&autoconnect=1`
