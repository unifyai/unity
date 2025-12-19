# Cursor Cloud Agent Dockerfile
# Uses shared system deps script to stay in sync with production Dockerfile.

FROM python:3.11-slim

# Install system dependencies from shared script (minimal set for testing)
COPY scripts/install-system-deps.sh /tmp/
RUN chmod +x /tmp/install-system-deps.sh && /tmp/install-system-deps.sh --minimal

# Set locale environment
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/

# Set up non-root user for Cursor Cloud Agent
RUN useradd -ms /bin/bash ubuntu && \
    echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu

ENV PATH="/home/ubuntu/.local/bin:${PATH}"
