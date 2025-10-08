Screen Share Manager Sandbox
============================

This sandbox lets you experiment with the `ScreenShareManager` in isolation by streaming a window from your screen, providing voice or text input to simulate user turns, and logging the captioned detected events in a Unify project. Details on how the manager works can be found in the base class that lives in `unity/screen_share_manager/screen_share_manager.py`

Running the sandbox
-------------------

1.  **Prerequisite**: [Install Docker Desktop](https://www.docker.com/products/docker-desktop/) for your operating system.

2.  **Start the Redis Container**: Open your terminal and run the following single command:

    ```bash
    docker run -d --name unity-redis -p 6379:6379 redis
    ```

    *   `-d`: Runs the container in "detached" mode (in the background).
    *   `--name unity-redis`: Gives the container a memorable name, making it easy to manage.
    *   `-p 6379:6379`: Maps port `6379` on your local machine to port `6379` inside the container. The code defaults to connecting to this port.
    *   `redis`: Tells Docker to download and run the official Redis image.
3. **Install required libraries** 
    ```bash
    pip install mss redis numpy Pillow opencv-python
    ```
4. **Setup environment variables** 
    | Variable Name | Requirement | Description |
    | :--- | :--- | :--- |
    | **`UNIFY_KEY`** | **Required** | This is essential. The `ScreenShareManager`'s core logic relies on calling the Unify client to analyze the user turn and generate descriptive captions, and log the transcripts in the Unify project. |
    | **`DEEPGRAM_API_KEY`** | **Required for `--voice`** | Only needed if you want to use voice input. The sandbox uses Deepgram for real-time speech-to-text transcription. |
    | **`CARTESIA_API_KEY`** | **Required for `--voice`** | Only needed if you want to use voice output. The sandbox uses Cartesia to generate the text-to-speech voice that confirms "Analysis complete." |
    | **`REDIS_HOST`** | Optional | Defaults to `localhost`. You only need to set this if your Redis server is running on a different machine or inside a specific Docker network. |
    | **`REDIS_PORT`** | Optional | Defaults to `6379`. You only need to set this if your Redis server is configured to listen on a non-standard port. |

5.  **Run the sandbox**:
    ```bash
    # Example usage after getting window coordinates
    python -m sandboxes.screen_share_manager.sandbox --x 100 --y 150 --width 1280 --height 720

    # The same, but with voice input/output enabled
    python -m sandboxes.screen_share_manager.sandbox --x 100 --y 150 --width 1280 --height 720 --voice
    ```

CLI flags
---------
This sandbox re-uses the common helper in `sandboxes/utils.py`, so it shares the standard options in addition to its own:

```
# Sandbox-specific flags
--x                 The x-coordinate of the top-left corner of the capture area. (Required)
--y                 The y-coordinate of the top-left corner of the capture area. (Required)
--width             The width of the capture area. (Required)
--height            The height of the capture area. (Required)
--fps               Frames per second for screen capture. (Default: 5)

# Standard flags
--voice / -v        Enable voice capture (Deepgram) + TTS playback (Cartesia)
--debug / -d        Show full reasoning steps of every tool-loop
--traced / -t       Wrap manager calls with unify.traced for detailed logs
--project_name / -p Name of the Unify project/context (default: "Sandbox")
--overwrite / -o    Delete any existing data for the chosen project before start
--project_version   Roll back to a specific project commit (int index)
--log_in_terminal   Stream logs to the terminal in addition to writing file logs
```

Interactive commands inside the REPL
------------------------------------
Once the sandbox starts, the screen capture will begin, and you can issue commands:

*   `<your message>` - Type any text to simulate a user utterance.
*   `r` - (Voice mode only) Press 'r' then Enter to record a voice utterance. Press Enter again to stop recording.
*   `help` | `h` - Show the help message.
*   `quit` | `exit` - Stop the screen capture and exit the sandbox.

Your utterance is sent for background processing immediately. Results from the analysis will appear asynchronously in the terminal as they are logged in the Unify project.

Logging and debugging
---------------------
*   Logged events can be seen in the Unify console under the selected project and context defaults to `Sandbox/Assistant/Transcripts` 
*   Running logs are written to `.logs_screen_share_sandbox.txt` (overwritten each run). Pass `--log_in_terminal` to also stream logs to the terminal.
