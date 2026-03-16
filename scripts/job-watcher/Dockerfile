FROM python:3.12-slim

ARG GITHUB_TOKEN
ARG BRANCH=main

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY scripts/job-watcher/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN DEP_BRANCH=$([ "$BRANCH" = "main" ] && echo "main" || echo "staging") && \
    git clone --depth 1 --branch $DEP_BRANCH https://${GITHUB_TOKEN}@github.com/unifyai/unify.git /tmp/unify && \
    pip install --no-cache-dir /tmp/unify && \
    rm -rf /tmp/unify

COPY unity/conversation_manager/assistant_jobs_api.py .
COPY scripts/job-watcher/watcher.py .

CMD ["kopf", "run", "watcher.py", "--liveness=http://0.0.0.0:8080/healthz", "--verbose"]
