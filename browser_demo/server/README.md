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

`docker run -it -p 6080:6080 -v $(pwd):/workspace --device=/dev/video10 --group-add video --rm bu_test bash`
