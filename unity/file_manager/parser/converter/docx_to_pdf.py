from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import os
import time
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .base import ConversionResult, PathLike


class DocxToPdfConverter:
    input_exts = {".doc", ".docx"}
    output_ext = ".pdf"

    # Global lock to serialize conversions for safety across threads/processes
    SOFFICE_LOCK = threading.Lock()

    def __init__(
        self,
        *,
        forced_backend: Optional[str] = None,
        overwrite: bool = False,
        logger: Optional[logging.Logger] = None,
        timeout_seconds: int = 300,
        allow_parallel_soffice: bool = False,
    ) -> None:
        self.forced_backend = forced_backend
        self.overwrite = overwrite
        self.log = logger or logging.getLogger(__name__)
        self.timeout_seconds = timeout_seconds
        # Some environments (e.g., WSL, CI) do NOT tolerate concurrent soffice invocations reliably
        self.allow_parallel_soffice = allow_parallel_soffice

    # ----- Public API -----

    def convert(
        self,
        src: PathLike,
        dst: Optional[PathLike] = None,
    ) -> ConversionResult:
        """Convert a .doc/.docx file to .pdf.

        If dst is omitted, the output is written as <src.stem>.pdf next to the source file.
        Honors the overwrite flag passed to the constructor; if a target already exists and overwrite=False,
        the method returns a successful result with backend="reuse" without performing conversion.
        """
        src_path = Path(src).expanduser().resolve()
        if not src_path.exists() or not src_path.is_file():
            return ConversionResult(
                False,
                src_path,
                None,
                "none",
                f"Source not found: {src_path}",
            )
        if src_path.suffix.lower() not in self.input_exts:
            return ConversionResult(
                False,
                src_path,
                None,
                "none",
                f"Unsupported extension: {src_path.suffix}",
            )

        dst_path = self._resolve_output_for_file(
            src_path,
            Path(dst).expanduser().resolve() if dst else None,
        )
        if dst_path.exists() and not self.overwrite:
            self.log.info(f"[doc->pdf] Reusing existing: {dst_path}")
            return ConversionResult(True, src_path, dst_path, "reuse", "exists")

        backend = self.forced_backend or self._pick_backend()
        self.log.info(
            f"[doc->pdf] Converting {src_path.name} -> {dst_path.name} (backend={backend})",
        )

        if backend == "win32com":
            ok, msg = self._convert_with_win32com(src_path, dst_path)
            if ok:
                return ConversionResult(True, src_path, dst_path, "win32com", "")
            if self._importable("docx2pdf"):
                ok, msg = self._convert_with_docx2pdf(src_path, dst_path)
                if ok:
                    return ConversionResult(True, src_path, dst_path, "docx2pdf", "")
            if self._have_soffice():
                ok, msg = self._convert_with_soffice(src_path, dst_path)
                if ok:
                    return ConversionResult(True, src_path, dst_path, "soffice", "")
            return ConversionResult(False, src_path, None, "win32com", msg)

        if backend == "docx2pdf":
            ok, msg = self._convert_with_docx2pdf(src_path, dst_path)
            if ok:
                return ConversionResult(True, src_path, dst_path, "docx2pdf", "")
            if self._have_soffice():
                ok, msg = self._convert_with_soffice(src_path, dst_path)
                if ok:
                    return ConversionResult(True, src_path, dst_path, "soffice", "")
            return ConversionResult(False, src_path, None, "docx2pdf", msg)

        if backend == "soffice":
            # Serialize unless explicitly allowed
            lock_ctx = (
                self.SOFFICE_LOCK if not self.allow_parallel_soffice else nullcontext()
            )
            with lock_ctx:
                ok, msg = self._convert_with_soffice(src_path, dst_path)
            return ConversionResult(
                ok,
                src_path,
                dst_path if ok else None,
                "soffice",
                msg if not ok else "",
            )

        return ConversionResult(
            False,
            src_path,
            None,
            "none",
            "No available backend (install Microsoft Word or LibreOffice).",
        )

    def convert_all(
        self,
        inputs: Iterable[PathLike],
        *,
        output_dir: Optional[PathLike] = None,
        parallel: bool = False,
    ) -> List[ConversionResult]:
        """Batch-convert .doc/.docx files to .pdf.

        If output_dir is set, outputs are written into it while preserving input filenames (with .pdf suffix).
        Unsupported inputs are returned as skipped results.

        Note on parallel=True:
            - LibreOffice (soffice) is generally OK to invoke concurrently on different files but may be heavy.
            - docx2pdf and win32com rely on Word; concurrent automation can misbehave.
            - For safety, we execute sequentially by default. If parallel=True, we best-effort parallelize only for
              the soffice path and keep others sequential.
        """
        indexed: list[tuple[int, Path]] = [
            (i, Path(src).expanduser().resolve()) for i, src in enumerate(inputs)
        ]
        backend = self.forced_backend or self._pick_backend()

        # Prepare output array preserving input order
        ordered_results: list[Optional[ConversionResult]] = [None] * len(indexed)

        # Helper to set a result by original index
        def _set(idx: int, res: ConversionResult) -> None:
            ordered_results[idx] = res

        # Decide strategy per item
        safe_items: list[tuple[int, Path]] = []
        sequential_items: list[tuple[int, Path]] = []
        for i, p in indexed:
            if p.suffix.lower() not in self.input_exts:
                _set(
                    i,
                    ConversionResult(
                        False,
                        p,
                        None,
                        "skip",
                        f"Unsupported: {p.suffix}",
                    ),
                )
                continue
            if parallel and backend == "soffice":
                safe_items.append((i, p))
            else:
                sequential_items.append((i, p))

        # Run sequential conversions first
        for i, p in sequential_items:
            dst = None
            if output_dir is not None:
                out_dir = Path(output_dir).expanduser().resolve()
                dst = out_dir.joinpath(p.name).with_suffix(self.output_ext)
            _set(i, self.convert(p, dst))

        # Run safe conversions potentially in parallel; still place results by original index
        if safe_items:
            if parallel:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def _do(job: tuple[int, Path]) -> tuple[int, ConversionResult]:
                    i, path = job
                    dst = None
                    if output_dir is not None:
                        out_dir = Path(output_dir).expanduser().resolve()
                        dst = out_dir.joinpath(path.name).with_suffix(self.output_ext)
                    return i, self.convert(path, dst)

                with ThreadPoolExecutor(max_workers=min(4, len(safe_items))) as ex:
                    futs = {ex.submit(_do, job): job[0] for job in safe_items}
                    for f in as_completed(futs):
                        try:
                            i, res = f.result()
                            _set(i, res)
                        except Exception as e:
                            i = futs[f]
                            _set(
                                i,
                                ConversionResult(
                                    False,
                                    safe_items[0][1],
                                    None,
                                    "soffice",
                                    f"Parallel error: {e}",
                                ),
                            )
            else:
                for i, p in safe_items:
                    dst = None
                    if output_dir is not None:
                        out_dir = Path(output_dir).expanduser().resolve()
                        dst = out_dir.joinpath(p.name).with_suffix(self.output_ext)
                    _set(i, self.convert(p, dst))

        # Replace any None with a generic failure (should not happen)
        return [
            (
                r
                if r is not None
                else ConversionResult(False, Path(""), None, "none", "unknown error")
            )
            for r in ordered_results
        ]

    # ----- Helpers -----

    def _importable(self, mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except Exception:
            return False

    def _have_soffice(self) -> Optional[str]:
        return shutil.which("soffice") or shutil.which("soffice.bin")

    def _pick_backend(self) -> str:
        sys = platform.system().lower()
        if sys == "windows":
            return (
                "win32com"
                if self._importable("win32com.client")
                else (
                    "docx2pdf"
                    if self._importable("docx2pdf")
                    else ("soffice" if self._have_soffice() else "none")
                )
            )
        if sys == "darwin":
            return (
                "docx2pdf"
                if self._importable("docx2pdf")
                else ("soffice" if self._have_soffice() else "none")
            )
        return "soffice" if self._have_soffice() else "none"

    def _resolve_output_for_file(
        self,
        src_file: Path,
        output_arg: Optional[Path],
    ) -> Path:
        if output_arg is None:
            return src_file.with_suffix(self.output_ext)
        if output_arg.suffix.lower() == self.output_ext:
            return output_arg
        return output_arg.joinpath(src_file.name).with_suffix(self.output_ext)

    # ----- Backend adapters -----

    def _convert_with_soffice(self, src: Path, dst: Path) -> Tuple[bool, str]:
        soffice = self._have_soffice()
        if not soffice:
            return False, "LibreOffice ('soffice') not found"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Use a unique user profile to avoid global profile locks when called concurrently
            unique_profile_dir = (
                dst.parent / f".soffice-profile-{os.getpid()}-{int(time.time()*1000)}"
            )
            try:
                unique_profile_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            cmd = [
                soffice,
                "--headless",
                "--nologo",
                "--nodefault",
                "--nolockcheck",
                "--nofirststartwizard",
                "--invisible",
                f"-env:UserInstallation=file://{unique_profile_dir.as_posix()}",
                "--convert-to",
                "pdf",
                "--outdir",
                str(dst.parent),
                str(src),
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout_seconds,
            )
            if proc.returncode != 0:
                return (
                    False,
                    f"soffice failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}",
                )
            # LibreOffice writes <src.stem>.pdf into outdir, not respecting dst name
            produced = src.with_suffix(self.output_ext).name
            produced_path = dst.parent / produced
            if produced_path.exists() and produced_path != dst:
                produced_path.replace(dst)
            return (dst.exists(), "" if dst.exists() else "Expected output not found")
        except subprocess.TimeoutExpired:
            return False, f"soffice timeout after {self.timeout_seconds}s"
        except Exception as e:
            return False, f"Exception: {e}"

    def _convert_with_docx2pdf(self, src: Path, dst: Path) -> Tuple[bool, str]:
        try:
            from docx2pdf import convert as docx2pdf_convert  # type: ignore
        except Exception as e:
            return False, f"docx2pdf not available: {e}"
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            before = {p.name for p in dst.parent.glob("*.pdf")}
            docx2pdf_convert(str(src), str(dst.parent))
            after = {p.name for p in dst.parent.glob("*.pdf")}
            new_files = list(after - before)
            candidates = [f"{src.stem}{self.output_ext}"] + new_files
            for name in candidates:
                p = dst.parent / name
                if p.exists():
                    if p != dst:
                        p.replace(dst)
                    return True, ""
            return False, "docx2pdf did not produce expected output"
        except Exception as e:
            return False, f"docx2pdf error: {e}"

    def _convert_with_win32com(self, src: Path, dst: Path) -> Tuple[bool, str]:
        try:
            import win32com.client  # type: ignore
            import pythoncom  # type: ignore
        except Exception as e:
            return False, f"pywin32 not available: {e}"
        try:
            pythoncom.CoInitialize()
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            wdFormatPDF = 17
            dst.parent.mkdir(parents=True, exist_ok=True)
            doc = word.Documents.Open(str(src), ReadOnly=True)
            doc.ExportAsFixedFormat(OutputFileName=str(dst), ExportFormat=wdFormatPDF)
            doc.Close(False)
            word.Quit()
            return (
                dst.exists(),
                "" if dst.exists() else "COM reported success but PDF not found",
            )
        except Exception as e:
            return False, f"Word COM error: {e}"
