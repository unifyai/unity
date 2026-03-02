#!/usr/bin/env bash
# setup_auth.sh — One-time setup for Unity GKE debug tools.
#
# Walks you through installing prerequisites and authenticating with GCP
# so that stream_logs.py can access live and historical job logs.
#
# Safe to re-run — it skips steps that are already complete.

set -euo pipefail

# Load .env from the project root.
if [[ -f ".env" ]]; then
    set -a
    source .env
    set +a
fi

# ─── Configuration ───────────────────────────────────────────────────────────
GCP_PROJECT="responsive-city-458413-a2"
GKE_CLUSTER="unity"
GKE_REGION="us-central1"

# ─── Colours ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
success() { echo -e "${GREEN}  ✓${NC} $*"; }
step()    { echo -e "\n${BOLD}── Step $1: $2 ──${NC}"; }

ISSUES=()

# ─── Step 1: gcloud CLI ─────────────────────────────────────────────────────
check_gcloud() {
    step 1 "Google Cloud CLI (gcloud)"

    if command -v gcloud >/dev/null 2>&1; then
        success "Installed — $(gcloud --version 2>/dev/null | head -1)"
    else
        warn "gcloud is not installed."
        echo ""
        echo "  Install options:"
        echo ""
        echo "  macOS (Homebrew — recommended):"
        echo "    brew install --cask google-cloud-sdk"
        echo ""
        echo "  macOS / Linux (manual installer):"
        echo "    curl https://sdk.cloud.google.com | bash"
        echo "    exec -l \$SHELL   # restart your shell"
        echo ""
        echo "  All platforms: https://cloud.google.com/sdk/docs/install"
        ISSUES+=("Install gcloud CLI")
    fi
}

# ─── Step 2: kubectl ─────────────────────────────────────────────────────────
check_kubectl() {
    step 2 "Kubernetes CLI (kubectl)"

    if command -v kubectl >/dev/null 2>&1; then
        local ver
        ver=$(kubectl version --client 2>/dev/null | head -1)
        success "Installed — ${ver}"
    else
        warn "kubectl is not installed."
        echo ""
        echo "  Install via gcloud (if gcloud is installed):"
        echo "    gcloud components install kubectl"
        echo ""
        echo "  Or via Homebrew:"
        echo "    brew install kubectl"
        echo ""
        echo "  Docs: https://kubernetes.io/docs/tasks/tools/"
        ISSUES+=("Install kubectl")
    fi
}

# ─── Step 3: GCP Authentication ──────────────────────────────────────────────
setup_gcp_auth() {
    step 3 "GCP Authentication"

    local current_account
    current_account=$(gcloud auth list --filter=status:ACTIVE \
        --format="value(account)" 2>/dev/null || echo "")

    if [[ -n "$current_account" ]]; then
        success "Authenticated as: ${current_account}"
        echo ""
        read -p "  Re-authenticate with a different account? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            info "Launching browser for authentication..."
            gcloud auth login
        fi
    else
        info "No active GCP account. Launching browser for authentication..."
        gcloud auth login
    fi

    # Set the project.
    info "Setting active project to ${GCP_PROJECT}..."
    gcloud config set project "$GCP_PROJECT" --quiet 2>/dev/null
    success "Project set to ${GCP_PROJECT}"
}

# ─── Step 4: GKE Cluster Credentials ────────────────────────────────────────
setup_gke_credentials() {
    step 4 "GKE Cluster Credentials"

    info "Fetching credentials for cluster '${GKE_CLUSTER}' (region ${GKE_REGION})..."

    if gcloud container clusters get-credentials "$GKE_CLUSTER" \
            --region "$GKE_REGION" \
            --project "$GCP_PROJECT" 2>&1; then
        success "Cluster credentials configured."
    else
        error "Failed to get cluster credentials."
        echo ""
        echo "  This usually means your GCP account lacks permissions."
        echo "  Ask a team admin to grant you access:"
        echo ""
        echo "    Project : ${GCP_PROJECT}"
        echo "    Role    : roles/container.viewer   (minimum for kubectl)"
        echo "    Role    : roles/logging.viewer      (for Cloud Logging)"
        echo ""
        ISSUES+=("Get GKE cluster access from a team admin")
        return 1
    fi

    # Verify connectivity.
    info "Verifying cluster connectivity..."
    if kubectl cluster-info --request-timeout=10s >/dev/null 2>&1; then
        success "Connected to GKE cluster."
    else
        warn "Could not verify cluster connectivity (credentials were still saved)."
    fi
}

# ─── Step 5: SHARED_UNIFY_KEY ────────────────────────────────────────────────
check_unify_key() {
    step 5 "Unify API Key (SHARED_UNIFY_KEY)"

    if [[ -n "${SHARED_UNIFY_KEY:-}" ]]; then
        success "SHARED_UNIFY_KEY is set."
    else
        warn "SHARED_UNIFY_KEY is not set."
        echo ""
        echo "  This key is needed to query the AssistantJobs project to"
        echo "  determine whether a job is currently running."
        echo ""
        echo "  Ask a team member for the shared Unify API key, then add"
        echo "  it to the project .env file:"
        echo ""
        echo "    # unity/.env"
        echo "    SHARED_UNIFY_KEY='your_key_here'"
        echo ""
        echo "  Both stream_logs.py and this script load from .env automatically."
        ISSUES+=("Set SHARED_UNIFY_KEY in the project .env file")
    fi
}

# ─── Step 6: Summary ─────────────────────────────────────────────────────────
print_summary() {
    step 6 "Summary"

    echo ""
    echo "  Component checks:"
    command -v gcloud  >/dev/null 2>&1 && echo -e "    ${GREEN}✓${NC} gcloud"  || echo -e "    ${RED}✗${NC} gcloud"
    command -v kubectl >/dev/null 2>&1 && echo -e "    ${GREEN}✓${NC} kubectl" || echo -e "    ${RED}✗${NC} kubectl"
    command -v curl    >/dev/null 2>&1 && echo -e "    ${GREEN}✓${NC} curl"    || echo -e "    ${RED}✗${NC} curl"

    local acct
    acct=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -1)
    [[ -n "$acct" ]]                   && echo -e "    ${GREEN}✓${NC} GCP auth ($acct)" || echo -e "    ${RED}✗${NC} GCP auth"

    kubectl cluster-info --request-timeout=5s >/dev/null 2>&1 \
        && echo -e "    ${GREEN}✓${NC} GKE cluster access" \
        || echo -e "    ${RED}✗${NC} GKE cluster access"

    [[ -n "${SHARED_UNIFY_KEY:-}" ]]   && echo -e "    ${GREEN}✓${NC} SHARED_UNIFY_KEY" || echo -e "    ${RED}✗${NC} SHARED_UNIFY_KEY"

    echo ""
    if [[ ${#ISSUES[@]} -eq 0 ]]; then
        echo -e "  ${GREEN}${BOLD}All checks passed!${NC} You're ready to use stream_logs.py."
        echo ""
        echo "  Example:"
        echo "    uv run scripts/dev/job_logs/stream_logs.py --job unity-2026-02-10-17-30-53-staging"
    else
        echo -e "  ${YELLOW}${BOLD}Remaining items (${#ISSUES[@]}):${NC}"
        for issue in "${ISSUES[@]}"; do
            echo -e "    ${YELLOW}→${NC} ${issue}"
        done
        echo ""
        echo "  Fix the above and re-run this script."
    fi
    echo ""
}

# ─── Main ────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${BOLD}════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  Unity GKE Debug Tools — Authentication Setup${NC}"
    echo -e "${BOLD}════════════════════════════════════════════════════════${NC}"

    check_gcloud
    check_kubectl

    # Only proceed to auth if the CLIs are available.
    if command -v gcloud >/dev/null 2>&1; then
        setup_gcp_auth
        setup_gke_credentials || true
    else
        warn "Skipping GCP auth (gcloud not installed)."
    fi

    check_unify_key
    print_summary
}

main "$@"
