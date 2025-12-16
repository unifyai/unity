#!/usr/bin/env zsh
# Unity test helper shell initialization
#
# Add this single line to your ~/.zshrc:
#   source ~/unity/tests/shell_init.zsh
#
# This sets up aliases and tab completions for all test helper scripts.

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

# ---- Aliases ----
alias parallel_run="$UNITY_TESTS_DIR/parallel_run.sh"
alias watch_tests="$UNITY_TESTS_DIR/watch_tests.sh"
alias attach="$UNITY_TESTS_DIR/attach.sh"
alias kill_failed="$UNITY_TESTS_DIR/kill_failed.sh"
alias kill_server="$UNITY_TESTS_DIR/kill_server.sh"
alias list_runs="$UNITY_TESTS_DIR/list_runs.sh"
alias monitor_resources="$UNITY_TESTS_DIR/monitor_resources.sh"
alias project_cleanup="$UNITY_TESTS_DIR/project_cleanup.sh"

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
        '-w[Wait for completion]' \
        '--wait[Wait for completion]' \
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
