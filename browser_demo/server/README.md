# Browser Use in Server

1. Build Dockerfile.

`docker build -t bu_test -f Dockerfile .`

2. Run Docker container, add `OPENAI_API_KEY` env variable through the env arg.

`docker run -it -p 6080:6080 --rm bu_test`

Note: add `-d` for detached mode

3. Live view the container browser locally at `http://localhost:6080/vnc.html?autoconnect=true&resize=scale&reconnect=true`

## Microphone Access with Custom Audio Input

1. Complete all steps above. In your own local browser, create a Google Meet with your `unify.ai` email.

2. Open the container browser through the link from step 3 in the last section.

3. Join the Google Meet URL in the container browser. Just create a name and no login required.

4. Once the user from the container browser is admitted into the Meet, input `play audio` in the interactive docker container CLI. You should be hearing the custom audio.

## Camera Access with Custom Video Input

Note: Unless you're on Ubuntu, else try this on VM

1. Start the VM instance (only 1 in the Unity project on GCP).

Note: Please stop the instance if not in use.

2. Clone the `unity` repo. Execute `cd unity/browser_demo/server`.

3. Run `bash host.sh` for installing prerequisites and initialising the camera device.

4. Build Docker image.

`docker build -t bu_test -f Dockerfile .`

5. Run the Docker container (works in both detached or interactive mode).

`docker run -it -p 6080:6080 -v $(pwd):/workspace --env OPENAI_API_KEY=<your-key> --device=/dev/video10 --group-add video --rm bu_test`

Note: Add `-d` for detached mode.

6. Get the external IP from GCP, and connect to the remote browser with the following link.

`http://<external-ip>:6080/vnc.html?autoconnect=true&resize=scale&reconnect=true`

7. Sign in to open a Google Meet (or join an existing one) to see the custom video running in a loop.
