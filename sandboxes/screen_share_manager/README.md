# Screen Share Manager Sandbox

This sandbox lets you experiment with the `ScreenShareManager` in isolation by demonstrating its **direct-control API**. It streams a window from your screen, allows you to provide voice or text input to simulate user turns, and prints the resulting annotated image handles.

The manager's core responsibility is to analyze screen and speech events and provide annotated `ImageHandle` objects for a consumer to use. This sandbox acts as that consumer.

### Demo
A demo video walking through a sample run can be found here: [Loom Video](https://www.loom.com/share/9040bec558804ef49ea0ba40dd7d5b9a?sid=19767836-0226-434a-bc9b-a465ea0eb974)

### Running the sandbox

#### 1. Install required libraries:
```bash
pip install unifyai mss Pillow opencv-python python-dotenv deepgram-sdk cartesia sounddevice
```

#### 2. Setup environment variables:
Create a `.env` file in the repository root or set the variables in your shell.

| Variable Name      | Requirement              | Description                                                                                                                              |
| :----------------- | :----------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **`UNIFY_KEY`**    | **Required**             | This is essential. The `ScreenShareManager` relies on the Unify client to analyze user turns and generate annotations for screen events. |
| **`DEEPGRAM_API_KEY`** | **Required for `--voice`** | Only needed if you want to use voice input. The sandbox uses Deepgram for real-time speech-to-text transcription.                         |
| **`CARTESIA_API_KEY`** | **Required for `--voice`** | Only needed if you want to use voice output. The sandbox uses Cartesia for text-to-speech voice confirmation.                          |

#### 3. Run the sandbox:
First, determine the coordinates of the window or screen region you want to capture. Then, run the sandbox with those coordinates.

```bash
# Example usage after getting window coordinates
python -m sandboxes.screen_share_manager.sandbox --x 100 --y 150 --width 1280 --height 720

# The same, but with voice input/output enabled
python -m sandboxes.screen_share_manager.sandbox --x 100 --y 150 --width 1280 --height 720 --voice
    
# Example with local image saving enabled 
python -m sandboxes.screen_share_manager.sandbox --x 100 --y 150 --width 1280 --height 720 --save-images 

# Example with initial context
python -m sandboxes.screen_share_manager.sandbox --x 100 --y 150 --width 1280 --height 720 --context "User is trying to log into a web portal."
```

### CLI flags

```
# Sandbox-specific flags
--x                 The x-coordinate of the top-left corner of the capture area. (Required)
--y                 The y-coordinate of the top-left corner of the capture area. (Required)
--width             The width of the capture area. (Required)
--height            The height of the capture area. (Required)
--fps               Frames per second for screen capture. (Default: 5)
--context           Initial session context to provide to the manager.
--save-images       Save annotated images locally to an 'images' folder.

# Standard flags
--voice / -v        Enable voice capture (Deepgram) + TTS playback (Cartesia)
--debug / -d        Show verbose tool logs (reasoning steps)
--project_name / -p Name of the Unify project/context (default: "Sandbox")
--log-in-terminal   Stream detailed logs to the terminal in addition to the log file.
```

### Interactive commands inside the REPL

Once the sandbox starts, screen capture will begin, and you can issue commands to simulate a user turn:

*   `<your message>` - Type any text and press Enter to simulate a user utterance. This will trigger the full detection and annotation pipeline.
*   `r` - (Voice mode only) Press 'r' then Enter to start recording. Speak your utterance and press Enter again to stop.
*   `help` | `h` - Show the help message.
*   `quit` | `exit` - Stop the screen capture and exit the sandbox.

Analysis results will appear in the terminal after the two-stage process completes for each turn.

### Logging and debugging

*   Analysis results are printed directly to the terminal as they are generated.
*   If you run with the `--save-images` flag, each annotated image from a turn will be saved to a local `images/` directory.
*   Running logs are written to `.logs_screen_share_sandbox.txt` (overwritten each run).