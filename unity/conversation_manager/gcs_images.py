"""Direct GCS upload, signed-URL generation, and download for images.

Uploads image bytes to a same-region GCS bucket and generates signed
URLs via local HMAC-SHA256 signing (no network round-trip).  The pod's
service account credentials at ``/secrets/key.json`` (or
``GOOGLE_APPLICATION_CREDENTIALS``) are used for both operations.
"""

from __future__ import annotations

import datetime
import logging
import os
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL import Image

from google.cloud import storage

LOGGER = logging.getLogger(__name__)

_BUCKET_NAME = os.environ.get("UNITY_IMAGE_BUCKET", "unity-screenshots")
_SIGNED_URL_EXPIRY_MINUTES = int(
    os.environ.get("UNITY_IMAGE_URL_EXPIRY_MINUTES", "60"),
)

_client_lock = threading.Lock()
_client: storage.Client | None = None


def _blob(bucket_name: str, object_path: str) -> storage.Blob:
    """Return a ``Blob`` handle, lazily initialising the GCS client."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = storage.Client()
    return _client.bucket(bucket_name).blob(object_path)


def _blob_from_uri(gcs_uri: str) -> storage.Blob:
    """Parse a ``gs://`` URI and return the corresponding ``Blob``."""
    prefix = "gs://"
    if not gcs_uri.startswith(prefix):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri!r}")
    remainder = gcs_uri[len(prefix) :]
    bucket_name, _, object_path = remainder.partition("/")
    if not object_path:
        raise ValueError(f"No object path in URI: {gcs_uri!r}")
    return _blob(bucket_name, object_path)


def upload_image(
    image_bytes: bytes,
    session_id: str,
    timestamp: str,
    source: str,
    content_type: str = "image/jpeg",
) -> str:
    """Upload image bytes to GCS and return the ``gs://`` URI.

    Parameters
    ----------
    image_bytes:
        Raw image bytes (JPEG, PNG, etc.).
    session_id:
        Unique identifier for the current session/job (used as the path
        prefix so all images from one session are grouped together).
    timestamp:
        ISO-formatted timestamp string used in the object name.
    source:
        Source label (``"user"``, ``"assistant"``, ``"webcam"``,
        ``"display"``, ``"file_ask"``, etc.).
    content_type:
        MIME type of the image (default ``"image/jpeg"``).

    Returns
    -------
    str
        A ``gs://`` URI pointing to the uploaded object.
    """
    ext = "jpg" if content_type == "image/jpeg" else content_type.split("/")[-1]
    safe_ts = timestamp.replace(":", "-")
    object_path = f"images/{session_id}/{safe_ts}_{source}.{ext}"
    blob = _blob(_BUCKET_NAME, object_path)
    blob.upload_from_string(image_bytes, content_type=content_type)
    return f"gs://{_BUCKET_NAME}/{object_path}"


def signed_url(
    gcs_uri: str,
    expiration_minutes: Optional[int] = None,
) -> str:
    """Generate a signed URL for a ``gs://`` URI using local crypto.

    This performs HMAC-SHA256 signing with the service account's private
    key -- zero network calls.  The result is a publicly-fetchable HTTPS
    URL valid for *expiration_minutes* (default: 60).
    """
    if expiration_minutes is None:
        expiration_minutes = _SIGNED_URL_EXPIRY_MINUTES

    return _blob_from_uri(gcs_uri).generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=expiration_minutes),
        method="GET",
    )


def download_bytes(gcs_uri: str) -> bytes:
    """Download the raw bytes of a GCS object identified by a ``gs://`` URI."""
    return _blob_from_uri(gcs_uri).download_as_bytes()


def download_image(gcs_uri: str) -> "Image.Image":
    """Download a GCS image and return it as a PIL Image.

    This is the primary way for the Actor to programmatically access
    screenshot and other image data for analysis, OCR, comparison, etc.

    Parameters
    ----------
    gcs_uri:
        A ``gs://`` URI pointing to an image object in GCS.

    Returns
    -------
    PIL.Image.Image
        The decoded image, ready for analysis.
    """
    import io

    import PIL.Image

    return PIL.Image.open(io.BytesIO(download_bytes(gcs_uri)))
