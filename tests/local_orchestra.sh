#!/usr/bin/env bash
# =============================================================================
# local_orchestra.sh - Manage a local Orchestra instance for testing
# =============================================================================
#
# This script starts a fully local Orchestra deployment using:
#   1. A Docker container running PostgreSQL with pgvector
#   2. The Orchestra FastAPI server from the adjacent ../orchestra repo
#
# This eliminates network latency and staging server bottlenecks during testing.
#
# Usage:
#   ./local_orchestra.sh start    # Start and wait for ready
#   ./local_orchestra.sh check    # Check if already running
#   ./local_orchestra.sh stop     # Stop local orchestra
#   ./local_orchestra.sh status   # Show status
#
# Environment:
#   ORCHESTRA_REPO_PATH   Override path to orchestra repo (default: ../orchestra)
#   ORCHESTRA_PORT        Override FastAPI port (default: 8000)
#   ORCHESTRA_DB_PORT     Override PostgreSQL port (default: 5432)
#
# On success, exports:
#   UNIFY_BASE_URL=http://127.0.0.1:8000/v0
#
set -euo pipefail

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
UNITY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"

# Configuration (can be overridden via environment)
ORCHESTRA_REPO_PATH="${ORCHESTRA_REPO_PATH:-$UNITY_ROOT/../orchestra}"
ORCHESTRA_PORT="${ORCHESTRA_PORT:-8000}"
ORCHESTRA_DB_PORT="${ORCHESTRA_DB_PORT:-5432}"
ORCHESTRA_DB_CONTAINER="unity-orchestra-db"
ORCHESTRA_SERVER_PIDFILE="/tmp/unity-orchestra-server.pid"
ORCHESTRA_SERVER_LOGFILE="/tmp/unity-orchestra-server.log"

# Local base URL that will be exported
LOCAL_ORCHESTRA_URL="http://127.0.0.1:${ORCHESTRA_PORT}/v0"

# Staging URL fallback
STAGING_URL="https://orchestra-staging-lz5fmz6i7q-ew.a.run.app/v0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# =============================================================================
# Prerequisite Checks
# =============================================================================

check_docker() {
  if ! command -v docker &>/dev/null; then
    log_error "Docker is not installed"
    return 1
  fi

  if ! docker info &>/dev/null; then
    log_error "Docker daemon is not running"
    return 1
  fi

  log_success "Docker is available"
  return 0
}

check_orchestra_repo() {
  local repo_path="$1"

  if [[ ! -d "$repo_path" ]]; then
    log_error "Orchestra repo not found at: $repo_path"
    return 1
  fi

  if [[ ! -f "$repo_path/pyproject.toml" ]]; then
    log_error "Orchestra repo appears incomplete (no pyproject.toml)"
    return 1
  fi

  if [[ ! -f "$repo_path/alembic.ini" ]]; then
    log_error "Orchestra repo missing alembic.ini"
    return 1
  fi

  log_success "Orchestra repo found at: $repo_path"
  return 0
}

check_poetry() {
  if ! command -v poetry &>/dev/null; then
    log_error "Poetry is not installed (required for orchestra)"
    return 1
  fi
  log_success "Poetry is available"
  return 0
}

# =============================================================================
# PostgreSQL Container Management
# =============================================================================

is_db_container_running() {
  docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${ORCHESTRA_DB_CONTAINER}$"
}

is_db_container_exists() {
  docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${ORCHESTRA_DB_CONTAINER}$"
}

is_compatible_db_running() {
  # Check if there's any pgvector container on the expected port that we can use
  # First check for any running postgres/pgvector container on our port
  local container
  container=$(docker ps --filter "publish=${ORCHESTRA_DB_PORT}" --format "{{.Names}}" 2>/dev/null | head -1)

  if [[ -n "$container" ]]; then
    # Check if we can connect to the orchestra database
    if docker exec "$container" pg_isready -U orchestra -d orchestra &>/dev/null; then
      log_success "Found compatible PostgreSQL container: $container"
      # Update the container name so other functions use it
      ORCHESTRA_DB_CONTAINER="$container"
      return 0
    fi
  fi
  return 1
}

start_db_container() {
  log_info "Starting PostgreSQL container with pgvector..."

  # First check if our named container is already running
  if is_db_container_running; then
    log_success "PostgreSQL container '$ORCHESTRA_DB_CONTAINER' already running"
    return 0
  fi

  # Check if there's another compatible container we can use
  if is_compatible_db_running; then
    return 0
  fi

  # If our container exists but not running, remove it and recreate
  if is_db_container_exists; then
    log_info "Removing stopped container..."
    docker rm "$ORCHESTRA_DB_CONTAINER" >/dev/null 2>&1 || true
  fi

  # Check if port is already in use by non-Docker process
  if lsof -i ":${ORCHESTRA_DB_PORT}" &>/dev/null; then
    # Port is in use - check if it's a PostgreSQL we can use
    log_warn "Port $ORCHESTRA_DB_PORT is already in use"

    # Try connecting to see if it's a PostgreSQL with our credentials
    if docker run --rm --network host pgvector/pgvector:pg15 \
         pg_isready -h localhost -p "$ORCHESTRA_DB_PORT" -U orchestra &>/dev/null 2>&1; then
      log_success "PostgreSQL already available on port $ORCHESTRA_DB_PORT"
      return 0
    fi

    # Try using psql to check if orchestra database exists
    if PGPASSWORD=orchestra psql -h localhost -p "$ORCHESTRA_DB_PORT" -U orchestra -d orchestra -c "SELECT 1" &>/dev/null 2>&1; then
      log_success "PostgreSQL with orchestra database available on port $ORCHESTRA_DB_PORT"
      return 0
    fi

    log_error "Port $ORCHESTRA_DB_PORT is in use but not by a compatible PostgreSQL"
    log_info "Try: docker stop orchestra-db (if running) or use ORCHESTRA_DB_PORT=5433"
    return 1
  fi

  # Calculate max_connections based on CPU cores (num_cores * 100)
  # This ensures enough connections for parallel test workers
  local num_cores
  if [[ "$(uname)" == "Darwin" ]]; then
    num_cores=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
  else
    num_cores=$(nproc 2>/dev/null || echo 4)
  fi
  local max_connections=$((num_cores * 100))
  log_info "Setting PostgreSQL max_connections=$max_connections (${num_cores} cores × 100)"

  # Start the container with increased max_connections
  docker run -d \
    --name "$ORCHESTRA_DB_CONTAINER" \
    -p "${ORCHESTRA_DB_PORT}:5432" \
    -e POSTGRES_PASSWORD=orchestra \
    -e POSTGRES_USER=orchestra \
    -e POSTGRES_DB=orchestra \
    pgvector/pgvector:pg15 \
    postgres -c max_connections="$max_connections" >/dev/null

  log_info "Waiting for PostgreSQL to be ready..."

  # Wait for PostgreSQL to be ready (max 30 seconds)
  local max_attempts=30
  local attempt=0
  while (( attempt < max_attempts )); do
    if docker exec "$ORCHESTRA_DB_CONTAINER" pg_isready -U orchestra &>/dev/null; then
      log_success "PostgreSQL is ready"
      return 0
    fi
    sleep 1
    ((attempt++))
  done

  log_error "PostgreSQL failed to start within 30 seconds"
  return 1
}

stop_db_container() {
  # Only stop containers we created (unity-orchestra-db)
  # Don't touch existing containers like orchestra-db
  local our_container="unity-orchestra-db"

  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${our_container}$"; then
    log_info "Stopping PostgreSQL container '$our_container'..."
    docker stop "$our_container" >/dev/null 2>&1 || true
    docker rm "$our_container" >/dev/null 2>&1 || true
    log_success "PostgreSQL container stopped"
  else
    log_info "No unity-specific PostgreSQL container to stop"
    log_info "(Leaving existing 'orchestra-db' container running)"
  fi
}

# =============================================================================
# Database Migrations and Seeding
# =============================================================================

run_migrations() {
  local repo_path="$1"

  log_info "Running database migrations..."

  cd "$repo_path"

  # Set environment variables for orchestra
  export ORCHESTRA_DB_HOST=localhost
  export ORCHESTRA_DB_PORT="$ORCHESTRA_DB_PORT"
  export ORCHESTRA_DB_USER=orchestra
  export ORCHESTRA_DB_PASS=orchestra
  export ORCHESTRA_DB_BASE=orchestra

  # Run migrations
  if poetry run alembic upgrade head 2>&1; then
    log_success "Migrations completed"
    return 0
  else
    log_error "Migrations failed"
    return 1
  fi
}

seed_test_user() {
  # Seed the database with a test user for Unity tests
  # Uses the API key from UNIFY_KEY env var (or falls back to a default)
  local test_user_id="unity-test-user-001"
  local test_api_key="${UNIFY_KEY:-unity-local-test-api-key}"
  local test_email="unity-test@debug.local"

  log_info "Checking if test user exists..."

  # Find the running postgres container
  local db_container
  db_container=$(docker ps --filter "publish=${ORCHESTRA_DB_PORT}" --format "{{.Names}}" 2>/dev/null | head -1)

  if [[ -z "$db_container" ]]; then
    log_error "No PostgreSQL container found"
    return 1
  fi

  # Check if user already exists
  local user_exists
  user_exists=$(docker exec "$db_container" psql -U orchestra -d orchestra -tAc \
    "SELECT 1 FROM users WHERE id = '$test_user_id'" 2>/dev/null || echo "")

  if [[ "$user_exists" == "1" ]]; then
    log_success "Test user already exists"
    return 0
  fi

  log_info "Creating test user for Unity tests..."

  # Create the test user and API key using docker exec
  docker exec "$db_container" psql -U orchestra -d orchestra -c "
-- Create billing user (users table)
INSERT INTO users (id, credits, stripe_customer_id, autorecharge, autorecharge_threshold, autorecharge_qty, store_prompts, frozen)
VALUES ('$test_user_id', 10000, null, false, 0, 0, true, false)
ON CONFLICT (id) DO NOTHING;

-- Create auth user record
INSERT INTO auth_user (id, email)
VALUES ('$test_user_id', '$test_email')
ON CONFLICT (id) DO NOTHING;

-- Create API key
INSERT INTO api_key (user_id, key)
VALUES ('$test_user_id', '$test_api_key')
ON CONFLICT (key) DO NOTHING;
" 2>&1

  if [[ $? -eq 0 ]]; then
    log_success "Test user created"
    log_info "Test API key: $test_api_key"
    return 0
  else
    log_error "Failed to create test user"
    return 1
  fi
}

seed_models_and_endpoints() {
  # Seed the database with models and endpoints required by Unity tests
  # This includes gpt-5.2@openai and other commonly used endpoints

  log_info "Seeding models and endpoints for Unity tests..."

  # Find the running postgres container
  local db_container
  db_container=$(docker ps --filter "publish=${ORCHESTRA_DB_PORT}" --format "{{.Names}}" 2>/dev/null | head -1)

  if [[ -z "$db_container" ]]; then
    log_error "No PostgreSQL container found"
    return 1
  fi

  # Check if models already seeded
  local model_exists
  model_exists=$(docker exec "$db_container" psql -U orchestra -d orchestra -tAc \
    "SELECT 1 FROM model WHERE mdl_code = 'gpt-5.2'" 2>/dev/null || echo "")

  if [[ "$model_exists" == "1" ]]; then
    log_success "Models and endpoints already seeded"
    return 0
  fi

  log_info "Creating models, providers, and endpoints..."

  # Seed all required data for Unity tests
  # Schema: provider(id, name, image_url, display_name)
  #         model(id, mdl_code, uploaded_at, task, active)
  #         endpoint(id, mdl_id, provider_id, created_at, active)
  docker exec "$db_container" psql -U orchestra -d orchestra -c "
-- Task and modality (required for models)
INSERT INTO modality (name) VALUES ('text_generation')
ON CONFLICT (name) DO NOTHING;

INSERT INTO task (name, modality) VALUES ('chat', 'text_generation')
ON CONFLICT (name) DO NOTHING;

-- Providers (id, name, image_url, display_name)
INSERT INTO provider (id, name, image_url, display_name) VALUES
  (1, 'openai', '', 'OpenAI'),
  (12, 'anthropic', '', 'Anthropic'),
  (36, 'vertex-ai', '', 'Google Vertex AI'),
  (37, 'deepseek', '', 'DeepSeek')
ON CONFLICT (id) DO NOTHING;

-- Models used by Unity tests (id, mdl_code, uploaded_at, task, active)
INSERT INTO model (id, mdl_code, uploaded_at, task, active) VALUES
  (100, 'gpt-5.2', NOW(), 'chat', true),
  (101, 'gpt-4o', NOW(), 'chat', true),
  (102, 'gpt-4o-mini', NOW(), 'chat', true),
  (103, 'gpt-3.5-turbo', NOW(), 'chat', true),
  (104, 'claude-3-5-sonnet', NOW(), 'chat', true),
  (105, 'claude-3-haiku', NOW(), 'chat', true),
  (106, 'claude-4.5-sonnet', NOW(), 'chat', true),
  (107, 'gemini-1.5-flash', NOW(), 'chat', true),
  (108, 'gemini-1.5-pro', NOW(), 'chat', true),
  (109, 'deepseek-v3', NOW(), 'chat', true)
ON CONFLICT (id) DO NOTHING;

-- Endpoints (id, mdl_id, provider_id, created_at, active)
INSERT INTO endpoint (id, mdl_id, provider_id, created_at, active) VALUES
  (100, 100, 1, NOW(), true),   -- gpt-5.2@openai
  (101, 101, 1, NOW(), true),   -- gpt-4o@openai
  (102, 102, 1, NOW(), true),   -- gpt-4o-mini@openai
  (103, 103, 1, NOW(), true),   -- gpt-3.5-turbo@openai
  (104, 104, 12, NOW(), true),  -- claude-3-5-sonnet@anthropic
  (105, 105, 12, NOW(), true),  -- claude-3-haiku@anthropic
  (106, 106, 12, NOW(), true),  -- claude-4.5-sonnet@anthropic
  (107, 107, 36, NOW(), true),  -- gemini-1.5-flash@vertex-ai
  (108, 108, 36, NOW(), true),  -- gemini-1.5-pro@vertex-ai
  (109, 109, 37, NOW(), true)   -- deepseek-v3@deepseek
ON CONFLICT (id) DO NOTHING;
" 2>&1

  if [[ $? -eq 0 ]]; then
    log_success "Models and endpoints seeded"
    return 0
  else
    log_error "Failed to seed models and endpoints"
    return 1
  fi
}

# Test API key that should be used with local orchestra
get_test_api_key() {
  echo "${UNIFY_KEY:-unity-local-test-api-key}"
}

# =============================================================================
# Orchestra Server Management
# =============================================================================

is_orchestra_server_running() {
  if [[ -f "$ORCHESTRA_SERVER_PIDFILE" ]]; then
    local pid
    pid=$(cat "$ORCHESTRA_SERVER_PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

is_orchestra_server_responsive() {
  curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${ORCHESTRA_PORT}/api/health" 2>/dev/null | grep -q "200\|404"
}

wait_for_server() {
  local max_attempts=60
  local attempt=0

  log_info "Waiting for Orchestra server to be ready..."

  while (( attempt < max_attempts )); do
    if curl -s "http://127.0.0.1:${ORCHESTRA_PORT}/v0" &>/dev/null || \
       curl -s "http://127.0.0.1:${ORCHESTRA_PORT}/docs" &>/dev/null; then
      log_success "Orchestra server is ready at $LOCAL_ORCHESTRA_URL"
      return 0
    fi
    sleep 1
    ((attempt++))
  done

  log_error "Orchestra server failed to start within 60 seconds"
  return 1
}

start_orchestra_server() {
  local repo_path="$1"

  log_info "Starting Orchestra FastAPI server..."

  if is_orchestra_server_running; then
    if wait_for_server; then
      log_success "Orchestra server already running"
      return 0
    else
      log_warn "Server process exists but not responsive, restarting..."
      stop_orchestra_server
    fi
  fi

  # Check if port is already in use
  if lsof -i ":${ORCHESTRA_PORT}" &>/dev/null; then
    # Check if it's already an orchestra server
    if wait_for_server; then
      log_success "Orchestra server already running on port $ORCHESTRA_PORT"
      return 0
    else
      log_error "Port $ORCHESTRA_PORT is in use by another process"
      return 1
    fi
  fi

  cd "$repo_path"

  # Set environment variables
  export ORCHESTRA_HOST=127.0.0.1
  export ORCHESTRA_PORT="$ORCHESTRA_PORT"
  export ORCHESTRA_DB_HOST=localhost
  export ORCHESTRA_DB_PORT="$ORCHESTRA_DB_PORT"
  export ORCHESTRA_DB_USER=orchestra
  export ORCHESTRA_DB_PASS=orchestra
  export ORCHESTRA_DB_BASE=orchestra
  export ORCHESTRA_RELOAD=false
  export ORCHESTRA_WORKERS_COUNT=1

  # Start server in background with workers matching CPU cores to handle parallel test load
  # Default uvicorn has 1 worker which can't handle 25+ concurrent test sessions
  local num_cores
  if [[ "$(uname)" == "Darwin" ]]; then
    num_cores=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
  else
    num_cores=$(nproc 2>/dev/null || echo 4)
  fi
  log_info "Starting Orchestra with $num_cores workers (matching CPU cores)"

  # Get the virtualenv python path - bypass poetry run which may interfere
  # with process management (poetry run can cause subprocess cleanup issues)
  local venv_python
  venv_python=$(poetry env info --executable 2>/dev/null)
  if [[ -z "$venv_python" || ! -x "$venv_python" ]]; then
    log_warn "Could not get virtualenv python path, falling back to poetry run"
    venv_python="poetry run python"
  else
    log_info "Using virtualenv python: $venv_python"
  fi

  # Start server directly with the virtualenv python (not via poetry run)
  # This avoids potential subprocess management issues with poetry
  ORCHESTRA_WORKERS_COUNT="$num_cores" nohup $venv_python -m orchestra > "$ORCHESTRA_SERVER_LOGFILE" 2>&1 &
  local pid=$!
  disown $pid 2>/dev/null || true
  echo "$pid" > "$ORCHESTRA_SERVER_PIDFILE"

  log_info "Orchestra server started with PID $pid"

  # Wait for server to be ready
  if wait_for_server; then
    return 0
  else
    log_error "Check logs at: $ORCHESTRA_SERVER_LOGFILE"
    return 1
  fi
}

stop_orchestra_server() {
  if [[ -f "$ORCHESTRA_SERVER_PIDFILE" ]]; then
    local pid
    pid=$(cat "$ORCHESTRA_SERVER_PIDFILE")

    if kill -0 "$pid" 2>/dev/null; then
      log_info "Stopping Orchestra server (PID $pid)..."
      kill "$pid" 2>/dev/null || true

      # Wait for graceful shutdown
      local attempt=0
      while (( attempt < 10 )); do
        if ! kill -0 "$pid" 2>/dev/null; then
          break
        fi
        sleep 1
        ((attempt++))
      done

      # Force kill if still running
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    fi

    rm -f "$ORCHESTRA_SERVER_PIDFILE"
  fi

  # Also kill any orphaned orchestra processes
  pkill -9 -f "python -m orchestra" 2>/dev/null || true

  log_success "Orchestra server stopped"
}

# =============================================================================
# Main Commands
# =============================================================================

cmd_start() {
  echo "=============================================="
  echo "Starting Local Orchestra for Unity Testing"
  echo "=============================================="
  echo ""

  # Check prerequisites
  if ! check_docker; then
    log_warn "Docker not available, falling back to staging URL"
    echo "export UNIFY_BASE_URL='$STAGING_URL'"
    return 1
  fi

  if ! check_orchestra_repo "$ORCHESTRA_REPO_PATH"; then
    log_warn "Orchestra repo not found, falling back to staging URL"
    echo "export UNIFY_BASE_URL='$STAGING_URL'"
    return 1
  fi

  if ! check_poetry; then
    log_warn "Poetry not available, falling back to staging URL"
    echo "export UNIFY_BASE_URL='$STAGING_URL'"
    return 1
  fi

  echo ""

  # Start components
  if ! start_db_container; then
    log_error "Failed to start database"
    echo "export UNIFY_BASE_URL='$STAGING_URL'"
    return 1
  fi

  if ! run_migrations "$ORCHESTRA_REPO_PATH"; then
    log_error "Failed to run migrations"
    echo "export UNIFY_BASE_URL='$STAGING_URL'"
    return 1
  fi

  if ! seed_test_user; then
    log_warn "Failed to seed test user (tests may fail without auth)"
  fi

  if ! seed_models_and_endpoints; then
    log_warn "Failed to seed models/endpoints (tests may fail without valid endpoints)"
  fi

  if ! start_orchestra_server "$ORCHESTRA_REPO_PATH"; then
    log_error "Failed to start Orchestra server"
    echo "export UNIFY_BASE_URL='$STAGING_URL'"
    return 1
  fi

  local test_api_key
  test_api_key=$(get_test_api_key)

  echo ""
  echo "=============================================="
  log_success "Local Orchestra is ready!"
  echo "=============================================="
  echo ""
  echo "To use in your shell:"
  echo "  export UNIFY_BASE_URL='$LOCAL_ORCHESTRA_URL'"
  echo "  export UNIFY_KEY='$test_api_key'"
  echo ""
  echo "Or source this script:"
  echo "  eval \"\$(./local_orchestra.sh)\""
  echo ""

  # Output the export commands for eval
  echo "export UNIFY_BASE_URL='$LOCAL_ORCHESTRA_URL'"
  echo "export UNIFY_KEY='$test_api_key'"

  return 0
}

cmd_stop() {
  echo "Stopping Local Orchestra..."
  echo ""

  stop_orchestra_server
  stop_db_container

  echo ""
  log_success "Local Orchestra stopped"
}

cmd_status() {
  echo "Local Orchestra Status"
  echo "======================"
  echo ""

  echo -n "Docker: "
  if check_docker 2>/dev/null; then
    echo -e "${GREEN}available${NC}"
  else
    echo -e "${RED}not available${NC}"
  fi

  echo -n "PostgreSQL Container: "
  if is_db_container_running; then
    echo -e "${GREEN}running ($ORCHESTRA_DB_CONTAINER)${NC}"
  elif is_compatible_db_running; then
    echo -e "${GREEN}running ($ORCHESTRA_DB_CONTAINER)${NC}"
  else
    echo -e "${RED}not running${NC}"
  fi

  echo -n "Orchestra Server: "
  if is_orchestra_server_running; then
    if wait_for_server 2>/dev/null; then
      echo -e "${GREEN}running and responsive${NC}"
    else
      echo -e "${YELLOW}running but not responsive${NC}"
    fi
  else
    echo -e "${RED}not running${NC}"
  fi

  echo ""
  echo "Configuration:"
  echo "  Orchestra Repo: $ORCHESTRA_REPO_PATH"
  echo "  FastAPI Port:   $ORCHESTRA_PORT"
  echo "  Database Port:  $ORCHESTRA_DB_PORT"
  echo "  Local URL:      $LOCAL_ORCHESTRA_URL"
  echo ""
}

cmd_check() {
  # Quick check if local orchestra is available - silent output
  # Just check if the server is responding, don't care about container names
  if curl -s "http://127.0.0.1:${ORCHESTRA_PORT}/v0" &>/dev/null || \
     curl -s "http://127.0.0.1:${ORCHESTRA_PORT}/docs" &>/dev/null; then
    echo "$LOCAL_ORCHESTRA_URL"
    return 0
  fi
  return 1
}

cmd_env() {
  # Output environment variables for local orchestra (if running)
  # Useful for: eval "$(./local_orchestra.sh env)"
  local test_api_key
  test_api_key=$(get_test_api_key)

  if cmd_check &>/dev/null; then
    echo "export UNIFY_BASE_URL='$LOCAL_ORCHESTRA_URL'"
    echo "export UNIFY_KEY='$test_api_key'"
  else
    # Fallback to staging
    echo "# Local orchestra not running, using staging"
    echo "export UNIFY_BASE_URL='$STAGING_URL'"
    # Don't output UNIFY_KEY - user needs to provide their own for staging
  fi
}

# =============================================================================
# Entry Point
# =============================================================================

main() {
  local cmd="${1:-start}"

  case "$cmd" in
    start)
      cmd_start
      ;;
    stop|--stop)
      cmd_stop
      ;;
    status|--status)
      cmd_status
      ;;
    check|--check)
      cmd_check
      ;;
    env|--env)
      cmd_env
      ;;
    -h|--help)
      echo "Usage: $0 [start|stop|status|check|env]"
      echo ""
      echo "Commands:"
      echo "  start    Start local orchestra (default)"
      echo "  stop     Stop local orchestra"
      echo "  status   Show status"
      echo "  check    Quick check if running (returns URL or exits 1)"
      echo "  env      Output environment variables for shell eval"
      echo ""
      echo "Environment:"
      echo "  ORCHESTRA_REPO_PATH  Path to orchestra repo (default: ../orchestra)"
      echo "  ORCHESTRA_PORT       FastAPI port (default: 8000)"
      echo "  ORCHESTRA_DB_PORT    PostgreSQL port (default: 5432)"
      echo ""
      echo "Quick usage:"
      echo "  eval \"\$($0 env)\"  # Set env vars if local orchestra running"
      ;;
    *)
      log_error "Unknown command: $cmd"
      echo "Run '$0 --help' for usage"
      exit 1
      ;;
  esac
}

main "$@"
