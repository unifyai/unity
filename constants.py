from datetime import datetime, timezone

SESSION_ID = datetime.now(timezone.utc).isoformat()
