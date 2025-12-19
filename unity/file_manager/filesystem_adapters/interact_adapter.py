from __future__ import annotations

from typing import Optional, Any, Dict, List
import base64
import json
import urllib.parse
import urllib.request
import time
import logging

from unity.settings import SETTINGS
from unity.file_manager.filesystem_adapters.base import BaseFileSystemAdapter
from unity.file_manager.types.filesystem import FileSystemCapabilities, FileReference


class InteractFileSystemAdapter(BaseFileSystemAdapter):
    """Adapter for Interact REST API (search + stream endpoints).

    Capabilities
    ------------
    - Read/list: implemented via `/search` and `/resource/stream` helpers
    - Rename/move: not available in public docs → left unimplemented

    Authentication
    --------------
    This adapter expects that Basic credentials (key:secret) are provided and
    can be used to call endpoints under the host supplied via ``api_base``.
    """

    def __init__(self, api_base: str, api_key: str, space: str):
        # All parameters can be overridden via SETTINGS
        self._base = (SETTINGS.INTERACT_API_BASE or api_base or "").rstrip("/")
        self._key = SETTINGS.INTERACT_KEY or api_key
        self._secret = SETTINGS.INTERACT_SECRET
        self._person_id = SETTINGS.INTERACT_PERSON_ID
        self._tenant = SETTINGS.INTERACT_TENANT
        self._space = space
        self._log = logging.getLogger(__name__)
        try:
            self._log.info(
                "Interact adapter init: base=%s tenant=%s person_id_set=%s",
                self._base,
                self._tenant,
                "yes" if bool(self._person_id) else "no",
            )
        except Exception:
            pass
        self._caps = FileSystemCapabilities(
            can_read=True,
            can_rename=False,
            can_move=False,
        )
        # Token cache
        self._access_token: Optional[str] = None
        self._token_expiry_ts: float = 0.0

    # ----------------------------- HTTP helpers ----------------------------- #
    def _token_headers(self) -> Dict[str, str]:
        hdrs = {"Content-Type": "application/json"}
        if self._tenant:
            hdrs["X-Tenant"] = self._tenant
        return hdrs

    def _auth_headers(self) -> Dict[str, str]:
        # Ensure we have a non-expired token
        self._ensure_token()
        hdrs = {"Authorization": f"Bearer {self._access_token}"}
        if self._tenant:
            hdrs["X-Tenant"] = self._tenant
        return hdrs

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Any = None,
        auth: bool = True,
    ) -> Any:
        url = f"{self._base}{path}"
        if params:
            q = urllib.parse.urlencode(params, doseq=False)
            self._log.info("Encoded params: %s", q)
            url = f"{url}?{q}"
        hdrs = dict(headers or {})
        hdrs.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        if auth:
            hdrs.update(self._auth_headers())
        try:
            # Avoid logging sensitive headers like Authorization
            safe_hdrs = {
                k: ("<redacted>" if k.lower() == "authorization" else v)
                for k, v in hdrs.items()
            }
            preview = None
            if data is not None and not isinstance(data, (bytes, bytearray)):
                try:
                    preview = json.dumps(data, ensure_ascii=False)[:500]
                except Exception:
                    preview = str(data)[:500]
            elif isinstance(data, (bytes, bytearray)):
                preview = f"<bytes:{len(data)}>"
            self._log.info(
                "HTTP %s %s params=%r headers=%r body=%r",
                method.upper(),
                url,
                params or {},
                safe_hdrs,
                preview,
            )
        except Exception as e:
            self._log.error("Error logging HTTP request: %s", e)
        req = urllib.request.Request(url, headers=hdrs, method=method.upper())
        body = None
        if data is not None:
            if isinstance(data, (bytes, bytearray)):
                body = bytes(data)
            elif isinstance(data, dict):
                ct = (hdrs.get("Content-Type") or "").lower()
                if "application/x-www-form-urlencoded" in ct:
                    body = urllib.parse.urlencode(data, doseq=True).encode("utf-8")
                else:
                    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            elif isinstance(data, str):
                body = data.encode("utf-8")
        with urllib.request.urlopen(
            req,
            data=body,
            timeout=30,
        ) as resp:  # nosec - env controlled
            status = getattr(resp, "status", None)
            if status is None:
                try:
                    status = resp.getcode()
                except Exception:
                    status = None
            data = resp.read()
        try:
            preview_resp = None
            try:
                preview_resp = json.loads(data.decode("utf-8"))
                # Keep info logging concise
                self._log.info(
                    "HTTP %s %s → %s json_keys=%s",
                    method.upper(),
                    url,
                    status,
                    list(preview_resp.keys()),
                )
            except Exception:
                preview_resp = (
                    data[:200] if isinstance(data, (bytes, bytearray)) else str(data)
                )
                self._log.info(
                    "HTTP %s %s → %s bytes=%s preview=%r",
                    method.upper(),
                    url,
                    status,
                    len(data) if isinstance(data, (bytes, bytearray)) else 0,
                    preview_resp,
                )
        except Exception:
            pass
        try:
            return json.loads(data.decode("utf-8"))
        except Exception:
            return data

    # ------------------------- Token management ------------------------- #
    def _ensure_token(self) -> None:
        now = time.time()
        if self._access_token and now < self._token_expiry_ts - 5:
            try:
                self._log.info(
                    "Token cached; expires_in=%.0fs",
                    self._token_expiry_ts - now,
                )
            except Exception:
                pass
            return
        try:
            self._log.info("Token missing/expired; fetching new token")
        except Exception:
            pass
        self._fetch_token()

    def _fetch_token(self) -> None:
        if not self._base or not self._person_id:
            # Cannot fetch without config
            self._access_token = None
            self._token_expiry_ts = 0.0
            return
        # Body as specified: grant_type, context, code=f"{Key}__{Secret}"
        code = f"{self._key}__{self._secret}" if self._secret else self._key
        body = {
            "grant_type": "authorization_code",
            "context": "KeySecret",
            "code": code,
        }
        try:
            self._log.info(
                "POST /token personid=%s body_keys=%s",
                self._person_id,
                list(body.keys()),
            )
        except Exception:
            pass
        payload = self._request(
            "POST",
            f"/token",
            params={"personid": self._person_id},
            headers=self._token_headers(),
            data=body,
            auth=False,
        )
        # Expected fields per docs: access_token, expires_in
        token = None
        expires_in = 0
        try:
            token = payload.get("access_token") if isinstance(payload, dict) else None
            expires_in = (
                int(payload.get("expires_in") or 0) if isinstance(payload, dict) else 0
            )
        except Exception:
            token = None
            expires_in = 0
        self._access_token = token
        self._token_expiry_ts = time.time() + max(
            0,
            expires_in or 1200,
        )  # default ~20m if missing
        try:
            self._log.info(
                "Token fetched: success=%s expires_in=%ss",
                bool(token),
                expires_in,
            )
        except Exception:
            pass

    @property
    def name(self) -> str:
        return f"Interact"

    @property
    def uri_name(self) -> str:
        return "interact"

    @property
    def capabilities(self) -> FileSystemCapabilities:
        return self._caps

    def _search(
        self,
        search_term: str,
        *,
        content_type: Optional[str] = None,
        excluded_types: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: Optional[str] = None,
        section_ids: Optional[List[int]] = None,
        hashtags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Call Interact's Search API and return the full payload.

        Mirrors documented query params:
          - searchTerm (required), type, excludedTypes, limit, offset, sortBy,
            sectionId (array of ints), hashtags (array of strings).

        Returns the raw response object which typically includes keys:
          SearchTerm, TotalResults, Results (list of result objects), ContentTypes, etc.
        """
        params: Dict[str, Any] = {
            "searchTerm": search_term,
            "limit": limit,
            "offset": offset,
        }
        if content_type:
            params["type"] = content_type
        if excluded_types:
            params["excludedTypes"] = excluded_types
        if sort_by:
            params["sortBy"] = sort_by
        if section_ids is not None:
            try:
                # Coerce to a list of ints even if a single int was provided
                if isinstance(section_ids, (list, tuple, set)):
                    params["sectionId"] = [int(x) for x in section_ids]
                else:
                    params["sectionId"] = [int(section_ids)]
            except Exception:
                # Fallback: best-effort string
                params["sectionId"] = [section_ids]
        if hashtags:
            params["hashtags"] = list(hashtags)

        try:
            self._log.info(
                "Search: term=%r type=%r excluded=%r limit=%s offset=%s sortBy=%r sectionId=%r hashtags=%r",
                search_term,
                content_type,
                excluded_types,
                limit,
                offset,
                sort_by,
                section_ids,
                hashtags,
            )
        except Exception:
            pass
        try:
            payload = self._request("GET", "/api/search", params=params)
            if isinstance(payload, dict):
                try:
                    self._log.info(
                        "Search response: TotalResults=%s page_count=%s",
                        payload.get("TotalResults"),
                        len(payload.get("Results") or []),
                    )
                except Exception:
                    pass
                return payload
        except Exception as e:
            try:
                self._log.info("Search error: %s", e)
            except Exception:
                pass
        return {
            "SearchTerm": search_term,
            "TotalResults": 0,
            "Results": [],
            "ContentTypes": [],
        }

    def stream_file(
        self,
        *,
        fileguid: str,
        filename: Optional[str] = None,
        area: Optional[str] = None,
    ) -> bytes:
        """Stream a file via /api/resource/stream using documented parameters.

        Parameters
        ----------
        fileguid : str
            GUID of the file to stream.
        filename : str | None
            Desired filename (with extension) for download response.
        area : str | None
            Area associated with the resource (e.g., "composer").
        """
        params: Dict[str, Any] = {"fileguid": fileguid}
        if filename:
            params["filename"] = filename
        if area:
            params["area"] = area
        try:
            self._log.info(
                "Stream: fileguid=%s filename=%r area=%r",
                fileguid,
                filename,
                area,
            )
            data = self._request("GET", "/api/resource/stream", params=params)
            if isinstance(data, (bytes, bytearray)):
                try:
                    self._log.info("Stream response: bytes=%s", len(data))
                except Exception:
                    pass
                return bytes(data)
            if isinstance(data, dict):
                blob = data.get("content") or data.get("data")
                if isinstance(blob, str):
                    try:
                        out = base64.b64decode(blob)
                        try:
                            self._log.info(
                                "Stream response: base64->bytes=%s",
                                len(out),
                            )
                        except Exception:
                            pass
                        return out
                    except Exception:
                        out = blob.encode("utf-8")
                        try:
                            self._log.info("Stream response: utf8-bytes=%s", len(out))
                        except Exception:
                            pass
                        return out
            return b""
        except Exception:
            return b""

    def iter_files(self, root: Optional[str] = None):
        # Treat search("*") as a generic listing when supported (wildcard)
        try:
            self._log.info("Iterating files via wildcard search '*'")
        except Exception:
            pass
        payload = self._search("*")
        items = payload.get("Results") or []
        try:
            self._log.info("Iter files: count=%s", len(items))
        except Exception:
            pass
        for it in items:
            # Try to extract a reasonable name/path; Interact content varies
            title = (
                it.get("Title")
                or it.get("title")
                or it.get("name")
                or it.get("Id")
                or "item"
            )
            sid = str(it.get("Id") or it.get("id") or it.get("assetId") or title)
            path = f"/{sid}"
            yield FileReference(
                path=path,
                name=str(title),
                provider=self.name,
                uri=f"{self.uri_name}://{sid}",
                size_bytes=None,
                modified_at=None,
                mime_type=None,
                extra={"raw": it},
            )

    def get_file(self, path: str) -> FileReference:
        sid = str(path).lstrip("/")
        # Best-effort: reuse search by id to obtain metadata
        meta = None
        try:
            self._log.info("Get file metadata via search: id=%s", sid)
            payload = self._search(sid, limit=10)
            results = payload.get("Results") or []
            meta = next(
                (r for r in results if str(r.get("Id") or r.get("id")) == sid),
                None,
            )
        except Exception:
            meta = None
        try:
            self._log.info("Get file: found_meta=%s", bool(meta))
        except Exception:
            pass
        return FileReference(
            path=f"/{sid}",
            name=str((meta or {}).get("Title") or (meta or {}).get("title") or sid),
            provider=self.name,
            uri=f"{self.uri_name}://{sid}",
            extra={"raw": meta} if meta else {},
        )

    def exists(self, path: str) -> bool:
        """Check if a file exists in the Interact system."""
        sid = str(path).lstrip("/")
        try:
            self._log.info("Check existence via search: id=%s", sid)
            payload = self._search(sid, limit=10)
            results = payload.get("Results") or []
            # Check if we can find a result matching this ID
            found = any(str(r.get("Id") or r.get("id")) == sid for r in results)
            self._log.info("Exists check: id=%s found=%s", sid, found)
            return found
        except Exception as e:
            self._log.info("Exists check failed: id=%s error=%s", sid, e)
            return False

    def list(self, root: Optional[str] = None) -> List[str]:
        """List all file paths in the Interact system."""
        try:
            return [ref.path.lstrip("/") for ref in self.iter_files(root)]
        except Exception:
            return []

    def open_bytes(self, path: str) -> bytes:
        # Prefer documented stream params; treat id value as fileguid best-effort
        sid = str(path).lstrip("/")
        try:
            self._log.info("Open bytes via stream: id=%s", sid)
        except Exception:
            pass
        return self.stream_file(fileguid=sid)

    def export_file(self, path: str, destination_dir: str) -> str:
        """Export (stream/download) a file from Interact system to local destination directory."""
        from pathlib import Path as _PathLib

        sid = str(path).lstrip("/")

        try:
            self._log.info(
                "Exporting file via stream: id=%s to %s",
                sid,
                destination_dir,
            )
        except Exception:
            pass

        # Stream file bytes from Interact API
        file_bytes = self.stream_file(fileguid=sid)

        if not file_bytes:
            raise FileNotFoundError(f"File not found or empty: {path}")

        # Create destination directory structure
        dest_dir = _PathLib(destination_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Preserve the original filename (use the ID as filename for Interact)
        # Try to get a better filename from metadata if available
        filename = path.lstrip("/")
        try:
            file_ref = self.get_file(path)
            if file_ref.name and file_ref.name != sid:
                filename = file_ref.name
        except Exception:
            pass

        dest_path = dest_dir / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Write bytes to destination file
        dest_path.write_bytes(file_bytes)

        try:
            self._log.info(
                "Exported file: %s -> %s (%d bytes)",
                sid,
                str(dest_path),
                len(file_bytes),
            )
        except Exception:
            pass

        return str(dest_path)

    def export_directory(self, path: str, destination_dir: str) -> List[str]:
        """Export (stream/download) all files from the Interact system.

        Streams files individually from the Interact API.
        """
        exported: List[str] = []

        try:
            self._log.info("Exporting directory: %s to %s", path, destination_dir)
        except Exception:
            pass

        try:
            for file_ref in self.iter_files(path):
                try:
                    exported_path = self.export_file(file_ref.path, destination_dir)
                    exported.append(exported_path)
                except Exception as e:
                    try:
                        self._log.warning(
                            "Failed to export file %s: %s",
                            file_ref.path,
                            e,
                        )
                    except Exception:
                        pass
                    continue
        except Exception as e:
            try:
                self._log.error("Failed to export directory %s: %s", path, e)
            except Exception:
                pass

        return exported

    def rename(
        self,
        path: str,
        new_name: str,
    ) -> FileReference:  # pragma: no cover - unimplemented
        raise NotImplementedError("Interact API does not expose rename in public docs")

    def move(
        self,
        path: str,
        new_parent_path: str,
    ) -> FileReference:  # pragma: no cover - unimplemented
        raise NotImplementedError("Interact API does not expose move in public docs")
