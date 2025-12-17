from __future__ import annotations

import logging
import platform
import shutil
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Optional

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
            self.log.info("[doc->pdf] Reusing existing: %s", dst_path)
            return ConversionResult(True, src_path, dst_path, "reuse", "exists")

        backend = self.forced_backend or self._pick_backend()
        self.log.info(
            "[doc->pdf] Converting %s -> %s (backend=%s)",
            src_path.name,
            dst_path.name,
            backend,
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
        """
        indexed: list[tuple[int, Path]] = [
            (i, Path(src).expanduser().resolve()) for i, src in enumerate(inputs)
        ]
        backend = self.forced_backend or self._pick_backend()

        ordered_results: list[Optional[ConversionResult]] = [None] * len(indexed)

        def _set(idx: int, res: ConversionResult) -> None:
            ordered_results[idx] = res

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

        for i, p in sequential_items:
            dst = None
            if output_dir is not None:
                out_dir = Path(output_dir).expanduser().resolve()
                dst = out_dir.joinpath(p.name).with_suffix(self.output_ext)
            _set(i, self.convert(p, dst))

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
        return "soffice" if self._have_soffice() else "none"

    def _resolve_output_for_file(self, src: Path, dst: Optional[Path]) -> Path:
        if dst is not None:
            return dst
        return src.with_suffix(self.output_ext)

    def _convert_with_docx2pdf(self, src: Path, dst: Path) -> tuple[bool, str]:
        try:
            from docx2pdf import convert  # type: ignore

            convert(str(src), str(dst))
            return (dst.exists(), "")
        except Exception as e:
            return (False, str(e))

    def _convert_with_win32com(self, src: Path, dst: Path) -> tuple[bool, str]:
        try:
            import win32com.client  # type: ignore

            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            doc = word.Documents.Open(str(src))
            doc.SaveAs(str(dst), FileFormat=17)  # 17 = wdFormatPDF
            doc.Close()
            word.Quit()
            return (dst.exists(), "")
        except Exception as e:
            return (False, str(e))

    def _convert_with_soffice(self, src: Path, dst: Path) -> tuple[bool, str]:
        soffice = self._have_soffice()
        if not soffice:
            return (False, "soffice not found on PATH")

        out_dir = dst.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # LibreOffice writes output to directory with same stem; we then move/rename if needed.
        cmd = [
            soffice,
            "--headless",
            "--nologo",
            "--nolockcheck",
            "--nodefault",
            "--norestore",
            "--convert-to",
            "pdf",
            "--outdir",
            str(out_dir),
            str(src),
        ]
        try:
            import subprocess

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            # LibreOffice output naming: <src.stem>.pdf
            produced = out_dir.joinpath(src.with_suffix(".pdf").name)
            if produced.exists() and produced != dst:
                try:
                    produced.replace(dst)
                except Exception:
                    # Best effort: leave produced in place if rename fails
                    pass
            ok = dst.exists() or produced.exists()
            if ok:
                return (True, "")
            return (
                False,
                (proc.stderr or proc.stdout or "soffice conversion failed").strip(),
            )
        except Exception as e:
            return (False, str(e))
