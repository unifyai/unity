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
# Branch logic mirrors CI: main→main, otherwise→staging
# This ensures staging deployments get staging branches (with latest fixes)
RUN DEP_BRANCH=$([ "$BRANCH" = "main" ] && echo "main" || echo "staging") && \
    git clone --depth 1 --branch $DEP_BRANCH https://github.com/unifyai/unify.git /unify && \
    git clone --depth 1 --branch $DEP_BRANCH https://github.com/unifyai/unillm.git /unillm

# Copy source and install unity with all dependencies
COPY . /app
RUN uv pip install --system --no-cache .

# Remove git credentials from config after install (security best practice)
RUN git config --global --unset url."https://${GITHUB_TOKEN}@github.com/".insteadOf

# Ensure entrypoint script is executable
RUN chmod +x /app/entrypoint.sh || true

# Build agent-service
WORKDIR /app/agent-service
RUN npm ci
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV UNIFY_KEY=${UNIFY_KEY}

RUN install -m 0755 /app/scripts/sandbox-dpkg /usr/local/bin/sandbox-dpkg

# Download the turn detector model files
# Set memory-efficient environment variables for model loading
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
RUN python unity/conversation_manager/medium_scripts/call.py download-files
RUN playwright install

# Set runtime environment variables for memory optimization
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Expose the ports that the applications use
# 8000: conversation manager, 6379: Redis, 3000: agent-service (Magnitude)
EXPOSE 8000 6379 3000

# Use Tini as init system to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Use bash to run the startup script
CMD ["/bin/bash", "/app/entrypoint.sh"]
