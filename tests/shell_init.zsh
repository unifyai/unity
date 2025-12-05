#!/usr/bin/env zsh
# Unity test helper shell initialization
#
# Add this single line to your ~/.zshrc:
#   source ~/unity/tests/shell_init.zsh
#
# This sets up aliases and tab completions for all test helper scripts.

# ---- Directory detection ----
UNITY_TESTS_DIR="${0:A:h}"  # Absolute path to directory containing this script

# ---- Aliases ----
alias parallel_run="$UNITY_TESTS_DIR/parallel_run.sh"
alias watch_tests="$UNITY_TESTS_DIR/watch_tests.sh"
alias attach="$UNITY_TESTS_DIR/attach.sh"
alias kill_failed="$UNITY_TESTS_DIR/kill_failed.sh"
alias kill_server="$UNITY_TESTS_DIR/kill_server.sh"
alias list_runs="$UNITY_TESTS_DIR/list_runs.sh"
alias monitor_resources="$UNITY_TESTS_DIR/monitor_resources.sh"

# ---- Completion: attach ----
# Completes tmux session names from the current terminal's socket
_unity_attach_complete() {
    local socket_name sessions

    # Derive socket name (same logic as attach.sh)
    local tty_id
    tty_id=$(tty 2>/dev/null)
    if [[ "$tty_id" == "not a tty" || -z "$tty_id" || ! "$tty_id" =~ ^/ ]]; then
        tty_id="pid$$"
    else
        tty_id=$(echo "$tty_id" | sed 's|/|_|g')
    fi
    socket_name="unity${tty_id}"

    # Get session names from tmux
    sessions=(${(f)"$(tmux -L "$socket_name" list-sessions -F '#{session_name}' 2>/dev/null)"})

    # Provide completions (properly quoted for spaces/special chars)
    _describe 'session' sessions
}
compdef _unity_attach_complete attach

# ---- Completion: parallel_run ----
# Completes test directories/files
_unity_parallel_run_complete() {
    local context state state_descr line
    typeset -A opt_args

    _arguments \
        '-t[Target test path]:test path:_files -W ~/unity/tests -g "*.py(/) test_*(/) "'  \
        '-n[Number of workers]:workers:(1 2 4 8 16)' \
        '-x[Stop on first failure]' \
        '--hierarchical[Run in hierarchical mode]' \
        '--flat[Run in flat mode]' \
        '--no-cache[Disable LLM cache]' \
        '--eval-only[Run only eval tests]' \
        '--symbolic-only[Run only symbolic tests]' \
        '--repeat[Repeat count]:count:(2 3 5 10)' \
        '-h[Show help]' \
        '--help[Show help]' \
        '*:test path:_files -W ~/unity/tests -g "*.py test_*/"'
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
