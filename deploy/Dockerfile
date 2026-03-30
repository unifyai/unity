# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Accept build arguments
ARG UNIFY_KEY
ARG GITHUB_TOKEN
ARG BRANCH=main

# Set working directory
WORKDIR /app

# Build environment setup
ENV DEBIAN_FRONTEND=noninteractive

# Install all system dependencies from shared script (full production set)
COPY scripts/install-system-deps.sh /tmp/
RUN chmod +x /tmp/install-system-deps.sh && /tmp/install-system-deps.sh && rm /tmp/install-system-deps.sh

# Set locale environment
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8


# Install Node.js & npm for agent-service
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs

# Install uv using official installation script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Configure git to use GITHUB_TOKEN for private repo authentication
RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

# Install PyTorch CPU-only first (smaller and faster for containers)
RUN uv pip install --system --no-cache torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Clone unify and unillm repos (no PyPI releases; pyproject.toml references ../unify and ../unillm)
# Branch logic mirrors CI: main→main, staging/preview→staging
# This keeps preview aligned with the existing staging dependency branches.
# DEP_REPOS_SHA is set at build time to the latest commit SHAs of unify+unillm;
# changing it busts the Docker layer cache so stale clones are never reused.
# The value is referenced in the RUN to ensure Docker includes it in the cache key.
ARG DEP_REPOS_SHA=unknown
RUN echo "dep_repos_sha=${DEP_REPOS_SHA}" && \
    DEP_BRANCH=$([ "$BRANCH" = "main" ] && echo "main" || echo "staging") && \
    git clone --depth 1 --branch $DEP_BRANCH https://github.com/unifyai/unify.git /unify && \
    git clone --depth 1 --branch $DEP_BRANCH https://github.com/unifyai/unillm.git /unillm

# Clone and build magnitude before COPY so this expensive layer is cached when
# only application source changes.  Placed outside /app to avoid being overwritten
# by COPY; symlinked back in after COPY for agent-service's file: references.
# --ignore-scripts skips the root postinstall (turbo run build) which requires bun.
# The committed lockfile only resolves platform-specific optional deps (sharp, baml, etc.)
# for the OS it was generated on, so we delete it and let npm resolve for the target platform.
RUN git clone --depth 1 --branch unity-modifications https://github.com/unifyai/magnitude.git /opt/magnitude
WORKDIR /opt/magnitude
RUN rm -f package-lock.json && npm install --ignore-scripts
RUN cd packages/magnitude-extract && npx tsup
RUN cd packages/magnitude-core && npx baml-cli generate && npx pkgroll
WORKDIR /app

# Remove git credentials after all git operations (security best practice)
RUN git config --global --unset url."https://${GITHUB_TOKEN}@github.com/".insteadOf

# Copy source and install unity with all dependencies
COPY . /app
# Strip [tool.uv.sources] — those are local-dev overrides that conflict with
# how dependencies are resolved inside the container (repos are at /unify, /unillm).
RUN sed -i '/^\[tool\.uv\.sources\]/,/^$/d' pyproject.toml
RUN uv pip install --system --no-cache .

# Link pre-built magnitude into the workspace (agent-service references ../magnitude)
RUN ln -sfn /opt/magnitude magnitude

# Ensure entrypoint script is executable
RUN chmod +x /app/deploy/entrypoint.sh || true

# Build agent-service
WORKDIR /app/agent-service
RUN npm ci && npx tsc --skipLibCheck
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV UNIFY_KEY=${UNIFY_KEY}

RUN install -m 0755 /app/scripts/sandbox-dpkg /usr/local/bin/sandbox-dpkg

# Download the turn detector model files to a fixed, user-agnostic path
# so the entrypoint can seed the emptyDir /tmp volume regardless of which user runs.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
RUN HF_HOME=/opt/hf-cache python unity/conversation_manager/medium_scripts/call.py download-files
RUN playwright install

# Set runtime environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Logging: Unity emoji logs to terminal, unify/unillm quiet (file traces only)
ENV UNITY_TERMINAL_LOG=true
ENV UNIFY_TERMINAL_LOG=false
ENV UNILLM_TERMINAL_LOG=false
ENV UNITY_LOG_DIR=/var/log/unity
ENV UNIFY_LOG_DIR=/var/log/unify
ENV UNILLM_LOG_DIR=/var/log/unillm
ENV MAGNITUDE_LOG_DIR=/var/log/magnitude
ENV MAGNITUDE_DEBUG=true

# Create non-root user for runtime
RUN useradd -m -u 1000 unity && \
    mkdir -p /var/log/unity /var/log/unify /var/log/unillm /var/log/magnitude /home/unity/.cache && \
    chown -R unity:unity /app /var/log/unity /var/log/unify /var/log/unillm /var/log/magnitude && \
    cp -r /root/.cache/ms-playwright /home/unity/.cache/ms-playwright && \
    chown -R unity:unity /home/unity/.cache

USER unity

# Expose the ports that the applications use
# 8000: conversation manager, 3000: agent-service (Magnitude)
EXPOSE 8000 3000

# Use Tini as init system to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Use bash to run the startup script
CMD ["/bin/bash", "/app/deploy/entrypoint.sh"]
