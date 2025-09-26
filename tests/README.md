## Run Python tests in parallel with tmux

This helper script launches one tmux session per Python file it finds and runs `pytest` for each file in its own window. It searches recursively and can also be restricted to specific folders or files.

When a session starts, it executes roughly:

```bash
export UNIFY_TESTS_RAND_PROJ=True
export UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True
source ~/unity/.unity/bin/activate
pytest <that_file.py>
```

## Live status and auto-close

- Status prefix: each tmux session name is prefixed with a typeable marker and emoji: `? ⏳` while the test runs, `o ✅` on success, or `x ❌` on failure. This makes it easy to tab-complete names in shells like zsh.
- Auto-close on success: sessions that pass are automatically killed about 10 seconds after completion. Failing sessions remain open for inspection.
- You can still attach before auto-close; you'll see the final message (e.g., `pytest exited with code: 0`) and a short notice that auto-close is scheduled.

## Install

Save the script at the repository root as a hidden file and make it executable:

```bash
curl -o .parallel_run.sh <paste-your-script-here-or-save-manually>
chmod +x .parallel_run.sh
```

## Requirements

- **tmux** and **pytest** must be installed (e.g., `brew install tmux`).
- **Virtualenv** is assumed to live at `~/unity/.unity/`. If yours differs, update the `source ~/unity/.unity/bin/activate` line inside the script.

## Basic usage

From the repository root, run:

```bash
./.parallel_run.sh
```

What happens:

- **Discovery**: Recursively finds all `*.py` files (excluding caches/venvs; see excludes below).
- **Sessions**: Creates one tmux session per file.
- **Window name**: The file’s basename without `.py`.
- **Session name**: Status-prefixed and derived from the file path, e.g., `tests/unit/test_math.py` → `? ⏳ tests-unit-test_math` (then `o ✅ tests-unit-test_math` or `x ❌ tests-unit-test_math`).

Common tmux actions:

```bash
tmux ls                                # list sessions
tmux attach -t <session-name>          # attach to a session
tmux switch-client -t <session-name>   # switch sessions (when already inside tmux)
```

## Targeting specific folders/files

Limit the search by passing directories and/or `.py` files. Examples:

```bash
# Only run files under a single folder
./.parallel_run.sh tests/integration

# Multiple roots
./.parallel_run.sh tests/unit tests/integration

# Specific files
./.parallel_run.sh tests/foo_test.py tests/bar_test.py

# Mix files and directories
./.parallel_run.sh tests/api tests/db/test_migrations.py
```

How it interprets arguments:

- **Directories**: Recursed (respecting excludes) to find `*.py`.
- **Files**: Run exactly as provided (no recursion).

## Defaults & conventions

- **Environment**: Exports `UNIFY_TESTS_RAND_PROJ=True` and `UNIFY_TESTS_DELETE_PROJ_ON_EXIT=True` inside each session so it works whether or not a tmux server is already running.
- **Virtualenv**: Assumes `~/unity/.unity/bin/activate`.
- **Excludes**: Skips directories: `.git`, `.hg`, `.svn`, `.venv`, `venv`, `.mypy_cache`, `.pytest_cache`, `__pycache__`, `.idea`, `.vscode`.
  - You can edit the `EXCLUDE_DIRS` array in the script to add/remove entries.
- **Names**:
  - Session: `<status-prefix> <relative-path-with-slashes-replaced-by-dashes>` (without `.py`). Example: `? ⏳ tests-unit-test_math` → `o ✅ tests-unit-test_math` or `x ❌ tests-unit-test_math`.
  - Window: `<filename-without-.py>`.
  - If a session name already exists, the script appends `-2`, `-3`, … to avoid collisions.

## Tips

- **Watch session statuses live**:

  ```bash
  watch -n 0.5 'tmux ls'
  ```

  As tests start, sessions show a `? ⏳` prefix. They flip to `o ✅` or `x ❌` when pytest exits. Successful sessions auto-close ~10s later.

- **Kill a session** once a test finishes:

  ```bash
  tmux kill-session -t <session-name>
  ```

  Note: sessions that pass auto-close within ~10 seconds; you typically only need to kill failing sessions.

- **Run in the background** (script exits immediately; sessions keep running):

  ```bash
  nohup ./.parallel_run.sh tests &>/dev/null &
  ```

- **See test output later**: just `tmux attach -t <session-name>` — pytest output stays in the window buffer.

## Troubleshooting

- **“tmux: command not found”**
  - Install tmux (e.g., `brew install tmux`, `apt-get install tmux`).

- **Virtualenv not found / wrong Python**
  - Update the activation line in the script:

    ```bash
    source /path/to/your/venv/bin/activate
    ```

- **No sessions created**
  - Ensure there are `.py` files under the provided paths and that excludes aren’t hiding your files.

- **Permission denied**
  - Make the script executable:

    ```bash
    chmod +x .parallel_run.sh
    ```

## Customization

Open `.parallel_run.sh` and tweak as needed:

- **`EXCLUDE_DIRS=( ... )`** — add/remove directories to skip.
- **`run_cmd()`** — change the command chain (e.g., add flags: `pytest -q -x`).
- **Session naming** — adjust `session_basename_for()` to your taste.

## Quick reference (tmux)

- Next/prev session (inside tmux):
  - Open the command prompt: `Ctrl-b :`
  - Type: `switch-client -n` (next) / `switch-client -p` (prev)
- List sessions: `tmux ls`
- Attach: `tmux attach -t <name>`
- Switch (inside tmux): `tmux switch-client -t <name>`
- Kill: `tmux kill-session -t <name>`

That’s it! Run it, list sessions, and jump into whichever test you want to watch.
