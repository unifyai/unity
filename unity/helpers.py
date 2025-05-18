import requests
from pathlib import Path

from unity.constants import PROJECT_ROOT, VENV_DIR


def _find_project_frame(start):
    """Return first frame in our project tree but *not* in the venv dir."""
    frame = start
    while frame is not None:
        p = Path(frame.f_code.co_filename).resolve()

        # True if p is inside PROJECT_ROOT (handles Py 3.8–3.10 gracefully)
        in_project = (
            p.is_relative_to(PROJECT_ROOT)
            if hasattr(p, "is_relative_to")
            else str(p).startswith(str(PROJECT_ROOT))
        )

        # Treat it as external if it lives in the venv folder
        in_venv = VENV_DIR in p.parents

        if in_project and not in_venv:
            return frame  # ← first “real” project frame
        frame = frame.f_back
    return None


def _handle_exceptions(response):
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}") from e
