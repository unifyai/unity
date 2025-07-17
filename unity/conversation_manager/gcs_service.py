import datetime
import hashlib
import logging
import os
import uuid
from typing import Tuple

from google.api_core import exceptions
from google.cloud import storage
from google.oauth2 import service_account


class GcsService:
    def __init__(self):
        self.credentials_path = os.getenv("ORCHESTRA_VERTEXAI_SERVICE_ACC_JSON")
        if not self.credentials_path:
            raise ValueError(
                "Missing GCP credentials key (ORCHESTRA_VERTEXAI_SERVICE_ACC_JSON)",
            )

        self.project_id = os.getenv("ORCHESTRA_VERTEXAI_PROJECT")
        if not self.project_id:
            raise ValueError("Missing GCP project ID (ORCHESTRA_VERTEXAI_PROJECT)")

        self.assistant_audio_bucket_name = os.getenv(
            "UNITY_ASSISTANT_AUDIO_BUCKET_NAME"
        )
        if not self.assistant_audio_bucket_name:
            raise ValueError(
                "Missing GCP assistant audio bucket name (UNITY_ASSISTANT_AUDIO_BUCKET_NAME)",
            )
        try:
            self.credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
            )
            self.storage_client = storage.Client(
                project=self.project_id,
                credentials=self.credentials,
            )
            self.bucket = self.storage_client.bucket(self.assistant_audio_bucket_name)
        except Exception as e:
            logging.error(f"Failed to initialize GCS client: {e}")
            raise

    def _generate_unique_filename(self, content: bytes) -> str:
        content_hash = hashlib.md5(content).hexdigest()
        unique_id = str(uuid.uuid4())[:8]
        return f"{content_hash}_{unique_id}"

    def upload_audio_file(
        self,
        file_content: bytes,
        user_id: str,
        assistant_id: str,
        content_type: str = "audio/wav",
    ) -> str:
        try:
            extension = content_type.split("/")[-1] if "/" in content_type else "wav"
            file_name = self._generate_unique_filename(file_content)
            object_path = f"{user_id}/{assistant_id}/{file_name}.{extension}"

            blob = self.bucket.blob(object_path)
            blob.upload_from_string(file_content, content_type=content_type)

            gcs_url = f"gs://{self.assistant_audio_bucket_name}/{object_path}"
            return gcs_url
        except exceptions.GoogleAPIError as e:
            logging.error(f"Failed to upload audio file to GCS: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in upload_audio_file: {str(e)}")
            raise


GCS_SERVICE = GcsService()
