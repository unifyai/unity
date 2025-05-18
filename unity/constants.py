import logging
from pathlib import Path
from datetime import datetime, timezone
import os  # NEW: for reading Anti-Captcha key

SESSION_ID = datetime.now(timezone.utc).isoformat()
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".unity"
LOGGER = logging.getLogger("unity")
ANTICAPTCHA_KEY = os.getenv("ANTICAPTCHA_KEY")  # NEW: Anti-Captcha API key
