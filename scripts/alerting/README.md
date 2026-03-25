# Discord Error Alerting

Sends system errors from Unity (and any future producer) to a Discord channel
via a shared GCP Pub/Sub topic. **Production only** — staging and preview
environments are silently skipped.

## Architecture

```
Producer (Unity / Adapters / Console)
  │  publish structured JSON
  ▼
Pub/Sub topic: discord-alerts
  │  push subscription
  ▼
Adapters Cloud Run: POST /discord/alert
  │  format embed, HTTP POST
  ▼
Discord Webhook → #your-channel
```

## Setup

### 1. Create the Discord webhook

1. Open Discord → Server Settings → Integrations → Webhooks
2. Create a webhook in your target channel
3. Copy the webhook URL

### 2. Provision the Pub/Sub topic and push subscription

```bash
cd scripts/alerting
chmod +x setup.sh
./setup.sh
```

This creates:
- Topic `discord-alerts`
- Push subscription `discord-alerts-push-sub`
  pointing at `POST /discord/alert` on the production adapters service

### 3. Set the webhook URL on the adapters Cloud Run service

```bash
gcloud run services update unity-adapters \
  --region us-central1 \
  --project responsive-city-458413-a2 \
  --set-env-vars DISCORD_ALERT_WEBHOOK_URL="https://discord.com/api/webhooks/..."
```

## Rate Limiting

Discord enforces a rate limit of 5 requests per 2 seconds per webhook.
If a burst of errors causes a 429, the adapters endpoint returns a non-200
status, so Pub/Sub nacks the message and retries with exponential backoff
(10s min, 600s max). No application-level rate limiting is needed.

## Message Schema

Producers publish JSON to the `discord-alerts` topic:

```json
{
  "title": "System Error — Assistant 'Alice' (ID: 25)",
  "description": "OOM prevention shutdown",
  "severity": "error",
  "source": "unity",
  "environment": "production",
  "timestamp": "2026-03-24T14:30:45Z",
  "fields": {
    "assistant_id": "25",
    "assistant_name": "Alice Smith",
    "job_name": "unity-2026-03-24-14-30-00",
    "user_name": "John Doe"
  },
  "traceback": "Traceback (most recent call last):\n  ..."
}
```

## Adding a New Producer

Any service can send Discord alerts by publishing to the `discord-alerts` topic.
The service only needs Pub/Sub publish access — no Discord webhook URL required.

```python
from google.cloud import pubsub_v1
import json

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path("responsive-city-458413-a2", "discord-alerts")
publisher.publish(
    topic_path,
    json.dumps({"title": "...", "description": "...", "severity": "error"}).encode(),
    source="my-service",
    severity="error",
)
```

## Extending

- **GCP Log Router sink**: Route Cloud Logging ERROR+ entries into the same
  `discord-alerts` topic for crash-safe coverage. No code changes needed.
- **Grafana alerts**: Configure a Discord contact point using the same webhook URL
  for aggregate anomaly alerts (idle pool depletion, latency spikes, etc.).
