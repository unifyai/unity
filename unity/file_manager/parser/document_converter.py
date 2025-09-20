from __future__ import annotations

"""
Office document converter (doc/docx → pdf) with extensible backend selection.

Backends supported out of the box:
- Windows: win32com (Word) → docx2pdf → LibreOffice (soffice)
- macOS:   docx2pdf → LibreOffice (soffice)
- Linux:   LibreOffice (soffice)

Design goals:
- Safe to use in parallel (no shared mutable state)
- Dependency injection-friendly (forced backend, overwrite policy, logging)
- Clear return semantics and robust error handling
"""

import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union


PathLike = Union[str, Path]


@dataclass
class ConversionResult:
    ok: bool
    src: Path
    dst: Optional[Path]
    backend: str
    message: str = ""


class DocumentConverter:
    def __init__(
        self,
        *,
        forced_backend: Optional[str] = None,
        overwrite: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.forced_backend = forced_backend
        self.overwrite = overwrite
        self.log = logger or logging.getLogger(__name__)

    # ---------- Public API ----------

    def convert(
        self,
        src: PathLike,
        dst: Optional[PathLike] = None,
    ) -> ConversionResult:
        """Convert a single .doc/.docx file to .pdf.

        If dst is None, writes <stem>.pdf next to src.
        Respects overwrite flag, otherwise reuses existing PDF.
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

        if src_path.suffix.lower() not in {".doc", ".docx"}:
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
            # fallbacks
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
        """Batch convert multiple files. Only .doc/.docx are attempted.

        output_dir: if provided, outputs mirror the file names into this directory.
        """
        srcs = [Path(p).expanduser().resolve() for p in inputs]

        # simple sequential by default; caller can parallelize externally if desired
        results: List[ConversionResult] = []
        for src in srcs:
            if src.suffix.lower() not in {".doc", ".docx"}:
                results.append(
                    ConversionResult(
                        False,
                        src,
                        None,
                        "skip",
                        f"Unsupported: {src.suffix}",
                    ),
                )
                continue
            dst = None
            if output_dir is not None:
                out_dir = Path(output_dir).expanduser().resolve()
                dst = out_dir.joinpath(src.name).with_suffix(".pdf")
            results.append(self.convert(src, dst))
        return results

    # ---------- Helpers ----------

    def _have_soffice(self) -> Optional[str]:
        return shutil.which("soffice") or shutil.which("soffice.bin")

    def _importable(self, mod: str) -> bool:
        try:
            __import__(mod)
            return True
        except Exception:
            return False

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
            return src_file.with_suffix(".pdf")
        if output_arg.suffix.lower() == ".pdf":
            return output_arg
        return output_arg.joinpath(src_file.name).with_suffix(".pdf")

    # ---------- Backend adapters ----------

    def _convert_with_soffice(self, src: Path, dst: Path) -> Tuple[bool, str]:
        soffice = self._have_soffice()
        if not soffice:
            return False, "LibreOffice ('soffice') not found"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            cmd = [
                soffice,
                "--headless",
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
            )
            if proc.returncode != 0:
                return (
                    False,
                    f"soffice failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}",
                )
            produced = src.with_suffix(".pdf").name
            produced_path = dst.parent / produced
            if produced_path.exists() and produced_path != dst:
                produced_path.replace(dst)
            return (dst.exists(), "" if dst.exists() else "Expected output not found")
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
            candidates = [f"{src.stem}.pdf"] + new_files
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
