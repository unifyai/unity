# Guided Learning Manager Sandbox

Interactive sandbox for testing the `GuidedLearningManager` keyframe capture system.

## Demo

Loom walkthrough: https://www.loom.com/share/6e46c5100f81477a8fa576ceee78681d

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Basic
python -m sandboxes.guided_learning_manager.sandbox

# With input listener for precise click/keyboard capture
python -m sandboxes.guided_learning_manager.sandbox --input-listener

# Custom region
python -m sandboxes.guided_learning_manager.sandbox --no-fullscreen --x 100 --y 100 --width 800 --height 800

# Actor execution mode (requires agent-service)
python -m sandboxes.guided_learning_manager.sandbox --enable-actor --execute-plan
```

## What It Does

1. Captures a region of your screen
2. Listens to your microphone for speech
3. Detects keyframes (visually significant moments)
4. Groups keyframes with transcribed speech into "steps"
5. Outputs `(transcript, keyframes[])` tuples

## Python API

```python
from unity.guided_learning_manager import GuidedLearningManager, GuidedLearningSettings
```

## CLI Arguments

### Screen Region

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-fullscreen` | off | Use custom region instead of fullscreen |
| `--x` | 0 | Left edge X (with --no-fullscreen) |
| `--y` | 0 | Top edge Y (with --no-fullscreen) |
| `--width` | screen | Capture width (with --no-fullscreen) |
| `--height` | screen | Capture height (with --no-fullscreen) |

### Capture & Selection Modes

| Argument | Default | Description |
|----------|---------|-------------|
| `--capture-mode` | `fps` | `fps`, `input_triggered`, or `hybrid` |
| `--selection-mode` | `llm` | `direct`, `algorithmic`, or `llm` |
| `--input-listener` | off | Enable pynput (auto-enables hybrid mode) |

### LLM Settings

| Argument | Default | Description |
|----------|---------|-------------|
| `--llm-model` | `gemini-2.5-flash@vertex-ai` | Vision model |
| `--llm-fps` | `0.5` | Frame buffer rate |
| `--llm-resolution` | `768x432` | Frame resolution |
| `--llm-max-frames` | `40` | Max frames per segment (applied after pre-filter) |
| `--no-prefilter` | off | Disable SSIM duplicate removal (LLM + DIRECT modes) |
| `--prefilter-threshold` | `0.98` | SSIM threshold |

### Input Listener (pynput)

| Argument | Default | Description |
|----------|---------|-------------|
| `--pynput-fps` | `10.0` | Frame buffer rate |
| `--pre-click-ms` | `100` | Capture frame ms before click |
| `--post-click-ms` | `300` | Capture frame ms after click |
| `--typing-interval` | `10` | Capture every N chars typed |
| `--show-input-events` | off | Print pynput events live |

### Audio Detection

| Argument | Default | Description |
|----------|---------|-------------|
| `--silence-threshold` | `300` | Audio level threshold |
| `--silence-duration` | `3.0` | Silence to end step (seconds) |
| `--min-speech` | `1.0` | Min speech duration to keep |
| `--countdown` | `5` | Countdown before start |

### Output & Debugging

| Argument | Default | Description |
|----------|---------|-------------|
| `--instrumentation-dir` | `captures/guided_learning/instrumentation` | Output directory |
| `--no-instrumentation` | off | Disable instrumentation |
| `--no-save-llm-frames` | off | Don't save LLM input frames |
| `--no-save-discarded` | off | Don't save prefilter-discarded frames |
| `--sandbox-debug` | off | Enable debug logging |

### Actor Integration

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable-actor` | off | Enable Actor integration for learning from demonstrations |
| `--execute-plan` | off | Execute learned plan during each step (requires agent-service) |
| `--debug` | off | Show full plan code instead of tree view |
| `--headless` | off | Run browser headless (no visible window) |

## Platform Setup

### macOS
```
System Settings → Privacy & Security → Accessibility → Enable app
System Settings → Privacy & Security → Input Monitoring → Enable app
```

### Windows
Run as Administrator for full keyboard/mouse capture.

### Linux
```bash
sudo usermod -aG input $USER
# For X11: xhost +local:
```

## Output Structure

```
captures/guided_learning/instrumentation/<session>/
├── steps/               # Per-step directories
│   ├── step_001/
│   │   ├── keyframes/           # Step's selected keyframes (PNG)
│   │   ├── llm_input_frames/    # Frames sent to LLM (JPG)
│   │   └── prefilter_discarded/ # Frames removed by SSIM
│   ├── step_002/
│   │   └── ...
│   └── ...
├── rejected_samples/    # Sample of rejected frames (if --save-rejected)
├── input_events.json    # Raw pynput events
├── report.json          # Full session report
├── summary.txt          # Human-readable summary
└── learned_plan.py      # Learned Actor plan (if --enable-actor)
```

## Actor Integration

The sandbox can integrate with the HierarchicalActor to learn workflows from demonstrations.

### Examples

```bash
# Basic Actor integration (learning mode)
python -m sandboxes.guided_learning_manager.sandbox --enable-actor

# With execution (requires agent-service running)
python -m sandboxes.guided_learning_manager.sandbox --enable-actor --execute-plan

# Debug mode (show full plan code)
python -m sandboxes.guided_learning_manager.sandbox --enable-actor --debug

# Combined with input listener for precise capture
python -m sandboxes.guided_learning_manager.sandbox --enable-actor --input-listener
```

### Modes

**Learning Mode (Default):**
- Mocked primitives (no real browser/API calls)
- Fast, safe learning
- Plan generation only
- Each demonstration step is processed via Actor interjection

**Execution Mode (`--execute-plan`):**
- Real primitives (actual browser/API calls)
- Requires agent-service running
- Plan generation + execution
- After each interjection, the sandbox waits until the Actor returns to
  `PAUSED_FOR_INTERJECTION` before listening for the next step

### Features

- **Live Plan Display**: After each step, shows a tree view of the generated plan with change indicators ([NEW], [MODIFIED])
- **Clarification Handling**: If the Actor needs clarification, you can respond via voice or keyboard
- **Session Summary**: Displays statistics about Actor learning (steps processed, functions generated, etc.)
- **Plan Persistence**: Automatically saves the learned plan to `learned_plan.py`

## Troubleshooting

**No keyframes captured?**
- Check your capture region coordinates
- Try `--input-listener` for precise capture
- Verify microphone permissions

**pynput not working?**
- Grant accessibility/input monitoring permissions
- Restart terminal after granting permissions
- Use `--show-input-events` to debug

**LLM selection failed?**
- Check `UNIFY_API_KEY` in `.env`
- Check network connectivity
