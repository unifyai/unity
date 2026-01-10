#!/usr/bin/env bash
#
# download_ci_logs.sh - Download CI log artifacts from GitHub Actions
#
# Usage:
#   ./logs/download_ci_logs.sh <url_or_run_id> [options]
#
# Examples:
#   # From artifact URL (copy from GitHub Actions UI)
#   ./logs/download_ci_logs.sh "https://github.com/unifyai/unity/actions/runs/20882540406/artifacts/5086119156"
#
#   # From run URL + pattern
#   ./logs/download_ci_logs.sh "https://github.com/unifyai/unity/actions/runs/20882540406" --pattern "function_manager"
#
#   # From run ID + pattern
#   ./logs/download_ci_logs.sh 20882540406 --pattern "function_manager"
#
#   # List all artifacts for a run
#   ./logs/download_ci_logs.sh 20882540406 --list
#
# Options:
#   --pattern <pattern>   Filter artifacts by name pattern (case-insensitive substring match)
#   --list                List available artifacts without downloading
#   --force               Re-download even if artifact already exists locally
#   --repo <owner/repo>   GitHub repository (default: unifyai/unity)
#   --help                Show this help message
#

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_LOGS_DIR="${SCRIPT_DIR}/ci"
DEFAULT_REPO="unifyai/unity"
MAX_WAIT_SECONDS=600  # 10 minutes max wait for artifacts
POLL_INTERVAL=10      # seconds between status checks

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

log_info() {
    echo -e "${BLUE}ℹ${NC} $*"
}

log_success() {
    echo -e "${GREEN}✓${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $*"
}

log_error() {
    echo -e "${RED}✗${NC} $*" >&2
}

show_help() {
    # Extract lines 3-26 (the usage documentation in the header comment)
    sed -n '3,26p' "$0" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

check_dependencies() {
    local missing=()

    if ! command -v gh &> /dev/null; then
        missing+=("gh (GitHub CLI)")
    fi

    if ! command -v jq &> /dev/null; then
        missing+=("jq")
    fi

    if ! command -v unzip &> /dev/null; then
        missing+=("unzip")
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing required dependencies:"
        for dep in "${missing[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "Install with:"
        echo "  brew install gh jq unzip  # macOS"
        echo "  apt install gh jq unzip   # Ubuntu/Debian"
        exit 1
    fi

    # Check gh auth
    if ! gh auth status &> /dev/null; then
        log_error "Not authenticated with GitHub CLI"
        echo "Run: gh auth login"
        exit 1
    fi
}

# Parse GitHub URL to extract run_id and optionally artifact_id
parse_url() {
    local input="$1"

    # Pattern: https://github.com/{owner}/{repo}/actions/runs/{run_id}/artifacts/{artifact_id}
    if [[ "$input" =~ actions/runs/([0-9]+)/artifacts/([0-9]+) ]]; then
        RUN_ID="${BASH_REMATCH[1]}"
        ARTIFACT_ID="${BASH_REMATCH[2]}"
        return 0
    fi

    # Pattern: https://github.com/{owner}/{repo}/actions/runs/{run_id}
    if [[ "$input" =~ actions/runs/([0-9]+) ]]; then
        RUN_ID="${BASH_REMATCH[1]}"
        ARTIFACT_ID=""
        return 0
    fi

    # Plain run ID (just numbers)
    if [[ "$input" =~ ^[0-9]+$ ]]; then
        RUN_ID="$input"
        ARTIFACT_ID=""
        return 0
    fi

    log_error "Could not parse input: $input"
    echo "Expected formats:"
    echo "  - Full artifact URL: https://github.com/owner/repo/actions/runs/123/artifacts/456"
    echo "  - Run URL: https://github.com/owner/repo/actions/runs/123"
    echo "  - Run ID: 123"
    return 1
}

# Get run status
get_run_status() {
    local repo="$1"
    local run_id="$2"

    gh api "repos/${repo}/actions/runs/${run_id}" --jq '.status' 2>/dev/null || echo "unknown"
}

# Wait for run to complete (or have artifacts available)
wait_for_artifacts() {
    local repo="$1"
    local run_id="$2"
    local pattern="${3:-}"

    local elapsed=0
    local status

    while [ $elapsed -lt $MAX_WAIT_SECONDS ]; do
        status=$(get_run_status "$repo" "$run_id")

        if [ "$status" = "completed" ]; then
            return 0
        fi

        # Check if target artifact is available even if run is in progress
        if [ -n "$pattern" ]; then
            local count
            count=$(gh api "repos/${repo}/actions/runs/${run_id}/artifacts?per_page=100" \
                --jq "[.artifacts[].name | select(. | ascii_downcase | contains(\"${pattern,,}\"))] | length" 2>/dev/null || echo "0")
            if [ "$count" -gt 0 ]; then
                log_info "Found matching artifact(s) while run is still in progress"
                return 0
            fi
        fi

        log_info "Run status: ${status} - waiting for artifacts... (${elapsed}s / ${MAX_WAIT_SECONDS}s)"
        sleep $POLL_INTERVAL
        elapsed=$((elapsed + POLL_INTERVAL))
    done

    log_warning "Timed out waiting for run to complete (status: $status)"
    log_info "Attempting to download available artifacts anyway..."
    return 0
}

# List all artifacts for a run
list_artifacts() {
    local repo="$1"
    local run_id="$2"
    local pattern="${3:-}"

    log_info "Listing artifacts for run ${run_id}..."

    local artifacts
    artifacts=$(gh api "repos/${repo}/actions/runs/${run_id}/artifacts?per_page=100" 2>/dev/null)

    if [ -z "$artifacts" ] || [ "$(echo "$artifacts" | jq '.total_count')" = "0" ]; then
        log_warning "No artifacts found for run ${run_id}"
        return 1
    fi

    echo ""
    echo -e "${CYAN}Available artifacts:${NC}"
    echo "────────────────────────────────────────────────────────────────"

    local filter_jq='.artifacts[]'
    if [ -n "$pattern" ]; then
        local pattern_lower="${pattern,,}"
        filter_jq=".artifacts[] | select(.name | ascii_downcase | contains(\"${pattern_lower}\"))"
    fi

    echo "$artifacts" | jq -r "${filter_jq} | \"  \(.id)\t\(.size_in_bytes / 1024 / 1024 | floor)MB\t\(.name)\"" | \
        while read -r line; do
            echo -e "$line"
        done

    echo "────────────────────────────────────────────────────────────────"
    echo ""

    local count
    count=$(echo "$artifacts" | jq "[${filter_jq}] | length")
    log_info "Total: ${count} artifact(s)"

    if [ -n "$pattern" ]; then
        log_info "Filter: '${pattern}'"
    fi
}

# Get artifact info by ID
get_artifact_info() {
    local repo="$1"
    local artifact_id="$2"

    gh api "repos/${repo}/actions/artifacts/${artifact_id}" 2>/dev/null
}

# Find artifacts by pattern
find_artifacts_by_pattern() {
    local repo="$1"
    local run_id="$2"
    local pattern="$3"

    local pattern_lower="${pattern,,}"

    gh api "repos/${repo}/actions/runs/${run_id}/artifacts?per_page=100" \
        --jq ".artifacts[] | select(.name | ascii_downcase | contains(\"${pattern_lower}\")) | {id: .id, name: .name, size: .size_in_bytes}" \
        2>/dev/null
}

# Download and extract artifact
download_artifact() {
    local repo="$1"
    local artifact_id="$2"
    local artifact_name="$3"
    local artifact_size="$4"
    local run_id="$5"
    local force="${6:-false}"

    local output_dir="${CI_LOGS_DIR}/${run_id}/${artifact_name}"
    local zip_file="${CI_LOGS_DIR}/${run_id}/${artifact_name}.zip"

    # Check if already downloaded
    if [ -d "$output_dir" ] && [ "$force" != "true" ]; then
        log_warning "Already downloaded: ${output_dir}"
        log_info "Use --force to re-download"
        return 0
    fi

    # Create directories
    mkdir -p "$(dirname "$zip_file")"

    # Calculate expected download time (rough estimate: 1MB/s)
    local size_mb=$((artifact_size / 1024 / 1024))
    local estimated_seconds=$((size_mb + 10))  # Add 10s buffer

    log_info "Downloading ${artifact_name} (${size_mb}MB)..."
    log_info "Estimated time: ~${estimated_seconds}s"

    # Download with progress
    local token
    token=$(gh auth token)

    # Use curl with progress bar for large files
    if ! curl -L \
        -H "Authorization: Bearer ${token}" \
        -H "Accept: application/vnd.github+json" \
        --progress-bar \
        -o "$zip_file" \
        "https://api.github.com/repos/${repo}/actions/artifacts/${artifact_id}/zip"; then
        log_error "Download failed"
        rm -f "$zip_file"
        return 1
    fi

    # Verify download
    if [ ! -f "$zip_file" ]; then
        log_error "Download failed - file not created"
        return 1
    fi

    local actual_size
    actual_size=$(stat -f%z "$zip_file" 2>/dev/null || stat -c%s "$zip_file" 2>/dev/null)

    if [ "$actual_size" -lt 1000 ]; then
        log_error "Download failed - file too small (${actual_size} bytes)"
        log_info "Contents: $(cat "$zip_file" | head -c 500)"
        rm -f "$zip_file"
        return 1
    fi

    log_success "Downloaded: ${zip_file} ($(echo "$actual_size / 1024 / 1024" | bc)MB)"

    # Extract
    log_info "Extracting to ${output_dir}..."
    mkdir -p "$output_dir"

    if ! unzip -q -o "$zip_file" -d "$output_dir"; then
        log_error "Extraction failed"
        return 1
    fi

    # Clean up zip file
    rm -f "$zip_file"

    log_success "Extracted to: ${output_dir}"

    # Update latest symlink
    local latest_link="${CI_LOGS_DIR}/latest"
    rm -f "$latest_link"
    ln -sf "${run_id}/${artifact_name}" "$latest_link"
    log_info "Updated symlink: logs/ci/latest → ${run_id}/${artifact_name}"

    return 0
}

# Show summary of downloaded logs
show_summary() {
    local output_dir="$1"

    echo ""
    echo -e "${GREEN}Download complete!${NC}"
    echo "────────────────────────────────────────────────────────────────"
    echo -e "📁 Location: ${CYAN}${output_dir}${NC}"
    echo ""

    # Show directory structure
    if command -v tree &> /dev/null; then
        tree -L 2 "$output_dir" 2>/dev/null | head -20
    else
        ls -la "$output_dir" | head -15
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo ""
    echo "Quick access:"
    echo "  cd ${output_dir}"
    echo ""

    # Check for pytest logs
    if [ -d "${output_dir}/pytest" ]; then
        local pytest_dir
        pytest_dir=$(ls -d "${output_dir}/pytest"/*/ 2>/dev/null | head -1)
        if [ -n "$pytest_dir" ]; then
            echo "Pytest logs:"
            echo "  ls ${pytest_dir}"
            echo ""

            # Show duration summary if available
            if [ -f "${pytest_dir}duration_summary.txt" ]; then
                echo "Duration summary:"
                echo "  cat ${pytest_dir}duration_summary.txt"
            fi
        fi
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

main() {
    local input=""
    local pattern=""
    local list_only=false
    local force=false
    local repo="$DEFAULT_REPO"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                show_help
                ;;
            --pattern|-p)
                pattern="$2"
                shift 2
                ;;
            --list|-l)
                list_only=true
                shift
                ;;
            --force|-f)
                force=true
                shift
                ;;
            --repo|-r)
                repo="$2"
                shift 2
                ;;
            -*)
                log_error "Unknown option: $1"
                echo "Use --help for usage"
                exit 1
                ;;
            *)
                if [ -z "$input" ]; then
                    input="$1"
                else
                    log_error "Unexpected argument: $1"
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # Validate input
    if [ -z "$input" ]; then
        log_error "No URL or run ID provided"
        echo ""
        show_help
    fi

    # Check dependencies
    check_dependencies

    # Parse input URL/ID
    RUN_ID=""
    ARTIFACT_ID=""
    if ! parse_url "$input"; then
        exit 1
    fi

    log_info "Repository: ${repo}"
    log_info "Run ID: ${RUN_ID}"

    if [ -n "$ARTIFACT_ID" ]; then
        log_info "Artifact ID: ${ARTIFACT_ID}"
    fi

    # Create ci logs directory
    mkdir -p "$CI_LOGS_DIR"

    # List mode
    if [ "$list_only" = true ]; then
        list_artifacts "$repo" "$RUN_ID" "$pattern"
        exit 0
    fi

    # If we have a direct artifact ID, download it
    if [ -n "$ARTIFACT_ID" ]; then
        local artifact_info
        artifact_info=$(get_artifact_info "$repo" "$ARTIFACT_ID")

        if [ -z "$artifact_info" ]; then
            log_error "Artifact ${ARTIFACT_ID} not found"
            exit 1
        fi

        local artifact_name artifact_size
        artifact_name=$(echo "$artifact_info" | jq -r '.name')
        artifact_size=$(echo "$artifact_info" | jq -r '.size_in_bytes')

        download_artifact "$repo" "$ARTIFACT_ID" "$artifact_name" "$artifact_size" "$RUN_ID" "$force"
        show_summary "${CI_LOGS_DIR}/${RUN_ID}/${artifact_name}"
        exit 0
    fi

    # Need pattern if no artifact ID
    if [ -z "$pattern" ]; then
        log_error "No artifact ID in URL and no --pattern specified"
        echo ""
        echo "Either:"
        echo "  1. Use a full artifact URL (includes artifact ID)"
        echo "  2. Specify --pattern to filter artifacts"
        echo "  3. Use --list to see available artifacts"
        exit 1
    fi

    # Wait for artifacts to be available
    wait_for_artifacts "$repo" "$RUN_ID" "$pattern"

    # Find matching artifacts
    local matching_artifacts
    matching_artifacts=$(find_artifacts_by_pattern "$repo" "$RUN_ID" "$pattern")

    if [ -z "$matching_artifacts" ]; then
        log_error "No artifacts matching pattern '${pattern}'"
        echo ""
        log_info "Available artifacts:"
        list_artifacts "$repo" "$RUN_ID" ""
        exit 1
    fi

    # Count matches
    local match_count
    match_count=$(echo "$matching_artifacts" | jq -s 'length')

    if [ "$match_count" -gt 1 ]; then
        log_warning "Multiple artifacts match pattern '${pattern}':"
        echo "$matching_artifacts" | jq -r '"  - \(.name) (\(.size / 1024 / 1024 | floor)MB)"'
        echo ""
        log_info "Downloading all ${match_count} matching artifacts..."
        echo ""
    fi

    # Download each matching artifact
    local downloaded=0
    while IFS= read -r artifact; do
        local artifact_id artifact_name artifact_size
        artifact_id=$(echo "$artifact" | jq -r '.id')
        artifact_name=$(echo "$artifact" | jq -r '.name')
        artifact_size=$(echo "$artifact" | jq -r '.size')

        if download_artifact "$repo" "$artifact_id" "$artifact_name" "$artifact_size" "$RUN_ID" "$force"; then
            downloaded=$((downloaded + 1))
        fi
        echo ""
    done < <(echo "$matching_artifacts" | jq -c '.')

    # Show summary
    if [ "$downloaded" -gt 0 ]; then
        if [ "$downloaded" -eq 1 ]; then
            # Single artifact - show detailed summary
            local artifact_name
            artifact_name=$(echo "$matching_artifacts" | jq -r '.name' | head -1)
            show_summary "${CI_LOGS_DIR}/${RUN_ID}/${artifact_name}"
        else
            # Multiple artifacts
            echo ""
            log_success "Downloaded ${downloaded} artifact(s) to: ${CI_LOGS_DIR}/${RUN_ID}/"
            echo ""
            ls -la "${CI_LOGS_DIR}/${RUN_ID}/"
        fi
    else
        log_error "No artifacts were downloaded"
        exit 1
    fi
}

main "$@"
