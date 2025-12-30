#!/usr/bin/env zsh
# Unity test helper shell initialization
#
# Add this single line to your ~/.zshrc:
#   source /path/to/your/unity/clone/tests/shell_init.zsh
#
# This sets up shell functions and tab completions for all test helper scripts.
#
# WORKTREE SUPPORT: Commands automatically detect the current git repository
# and use that repo's scripts. This means they work correctly in git worktrees
# (e.g., Cursor Background Agents) without any extra configuration.

# ---- Directory detection ----
UNITY_TESTS_DIR="${0:A:h}"  # Absolute path to directory containing this script

# ---- Socket name (captured at init time for completions) ----
# Must be computed here because `tty` doesn't work inside completion functions
_unity_tty_id=$(tty 2>/dev/null)
if [[ "$_unity_tty_id" == "not a tty" || -z "$_unity_tty_id" || ! "$_unity_tty_id" =~ ^/ ]]; then
    _UNITY_SOCKET="unity_default"
else
    _UNITY_SOCKET="unity${_unity_tty_id//\//_}"
fi
unset _unity_tty_id

# ---- Dynamic script resolver ----
# Returns the path to a test script, preferring the current git repo's version.
# This enables worktree support: when you're in a worktree, commands use that
# worktree's scripts (and thus test that worktree's code).
_unity_resolve_script() {
    local script_name="$1"
    local git_root
    git_root=$(git rev-parse --show-toplevel 2>/dev/null)
    if [[ -n "$git_root" && -x "$git_root/tests/$script_name" ]]; then
        echo "$git_root/tests/$script_name"
    else
        echo "$UNITY_TESTS_DIR/$script_name"
    fi
}

# ---- Shell functions (worktree-aware) ----
# These functions dynamically resolve to the current repo's scripts, enabling
# seamless operation in git worktrees without manual path adjustments.

parallel_run() {
    "$(_unity_resolve_script parallel_run.sh)" "$@"
}

parallel_cloud_run() {
    "$(_unity_resolve_script parallel_cloud_run.sh)" "$@"
}

watch_tests() {
    "$(_unity_resolve_script watch_tests.sh)" "$@"
}

attach() {
    "$(_unity_resolve_script attach.sh)" "$@"
}

kill_failed() {
    "$(_unity_resolve_script kill_failed.sh)" "$@"
}

kill_server() {
    "$(_unity_resolve_script kill_server.sh)" "$@"
}

list_runs() {
    "$(_unity_resolve_script list_runs.sh)" "$@"
}

monitor_resources() {
    "$(_unity_resolve_script monitor_resources.sh)" "$@"
}

project_cleanup() {
    "$(_unity_resolve_script project_cleanup.sh)" "$@"
}

orchestra() {
    "$(_unity_resolve_script orchestra.sh)" "$@"
}

# ---- Completion: attach ----
_unity_attach_complete() {
    local -a sessions
    sessions=(${(f)"$(tmux -L "$_UNITY_SOCKET" list-sessions -F '#{session_name}' 2>/dev/null)"})
    [[ -n "${sessions[*]}" ]] && compadd "${sessions[@]}"
}
compdef _unity_attach_complete attach

# ---- Completion: parallel_run ----
# Completes flags and test directories/files
_unity_parallel_run_complete() {
    _arguments \
        '-t[Timeout in seconds]:timeout:(60 120 300 600)' \
        '--timeout[Timeout in seconds]:timeout:(60 120 300 600)' \
        '-s[Serial mode (one session per file)]' \
        '--serial[Serial mode (one session per file)]' \
        '-j[Job limit]:jobs:(8 16 25 40 0)' \
        '--jobs[Job limit]:jobs:(8 16 25 40 0)' \
        '-m[Match filename pattern]:pattern:' \
        '--match[Match filename pattern]:pattern:' \
        '-e[Set environment variable]:var:' \
        '--env[Set environment variable]:var:' \
        '--tags[Add test tags]:tags:' \
        '--eval-only[Run only eval tests]' \
        '--symbolic-only[Run only symbolic tests]' \
        '--repeat[Repeat count]:count:(2 3 5 10)' \
        '-h[Show help]' \
        '--help[Show help]' \
        '*:test path:_files'
}
compdef _unity_parallel_run_complete parallel_run

# ---- Completion: parallel_cloud_run ----
# Completes --env flags and test directories/files
_unity_parallel_cloud_run_complete() {
    _arguments \
        '*--env[Set environment variable]:var:' \
        '*:test path:_files'
}
compdef _unity_parallel_cloud_run_complete parallel_cloud_run

# ---- Completion: list_runs ----
_unity_list_runs_complete() {
    _arguments \
        '--all[List sessions from all terminals]' \
        '-h[Show help]' \
        '--help[Show help]'
}
compdef _unity_list_runs_complete list_runs

# ---- Completion: kill_failed ----
_unity_kill_failed_complete() {
    _arguments \
        '--all[Kill failed sessions from all terminals]' \
        '-h[Show help]' \
        '--help[Show help]'
}
compdef _unity_kill_failed_complete kill_failed

# ---- Completion: project_cleanup ----
_unity_project_cleanup_complete() {
    _arguments \
        '--dry-run[Show matching projects without deleting]' \
        '-y[Skip confirmation prompt]' \
        '--yes[Skip confirmation prompt]' \
        '--shared-only[Only delete the shared UnityTests project]' \
        '--random-only[Only delete random UnityTests_* projects]' \
        '--prefix[Override prefix for random projects]:prefix:' \
        '-s[Use staging environment]' \
        '--staging[Use staging environment]' \
        '-p[Use production environment]' \
        '--production[Use production environment]' \
        '-h[Show help]' \
        '--help[Show help]'
}
compdef _unity_project_cleanup_complete project_cleanup
