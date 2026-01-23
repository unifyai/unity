from __future__ import annotations

"""
Local pytest fixtures and test-only stubs for the `test_file_parser/` suite.

Why this exists
--------------
This sub-suite has a few shared needs:
- Locating the canonical `tests/test_file_manager/sample/` files
- Creating tiny XLSX fixtures without optional third-party deps (no openpyxl)
- Testing ParseConfig hot-swapping via BackendRegistry's lazy dotted-path imports

We keep these helpers here (instead of `tests/` top-level utility modules) so:
- test helpers live next to the suite they support
- reuse is via fixtures (pytest-native) rather than ad-hoc imports
"""

from pathlib import Path
from textwrap import dedent
import zipfile
from typing import Optional, Sequence

import pytest

from unity.file_manager.file_parsers.types.backend import BaseFileParserBackend
from unity.file_manager.file_parsers.types.contracts import (
    FileParseMetadata,
    FileParseRequest,
    FileParseResult,
)
from unity.file_manager.file_parsers.types.formats import FileFormat, MimeType

# ---------------------------------------------------------------------------
# Sample file helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_dir() -> Path:
    """Return the canonical sample directory used by FileManager/FileParser tests."""
    d = Path(__file__).resolve().parents[1] / "sample"
    assert d.exists() and d.is_dir(), f"Missing sample dir: {d}"
    return d


@pytest.fixture()
def sample_file(sample_dir: Path):
    """Callable fixture: sample_file(name) -> Path."""

    def _sample_file(name: str) -> Path:
        p = sample_dir / name
        assert p.exists() and p.is_file(), f"Missing sample file: {p}"
        return p

    return _sample_file


# ---------------------------------------------------------------------------
# XLSX fixture generator (stdlib only; no optional deps)
# ---------------------------------------------------------------------------


@pytest.fixture()
def write_minimal_xlsx():
    """
    Callable fixture: write_minimal_xlsx(path, sheets=[(sheet_name, rows), ...])

    Produces a minimal XLSX (OpenXML) file suitable for end-to-end parsing tests without
    pulling in optional deps like openpyxl.
    """

    def _write_minimal_xlsx(
        path: Path,
        *,
        sheets: list[tuple[str, list[list[str]]]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ns_main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

        def _cell_ref(col_ix: int, row_ix: int) -> str:
            col = ""
            x = col_ix + 1
            while x:
                x, r = divmod(x - 1, 26)
                col = chr(ord("A") + r) + col
            return f"{col}{row_ix+1}"

        def _sheet_xml(rows: list[list[str]]) -> str:
            row_xml = []
            for r_ix, row in enumerate(rows):
                cells = []
                for c_ix, val in enumerate(row):
                    r = _cell_ref(c_ix, r_ix)
                    v = val or ""
                    if v.startswith("="):
                        # Store formula with cached numeric 0 (backends may or may not evaluate).
                        cells.append(f'<c r="{r}"><f>{v[1:]}</f><v>0</v></c>')
                    else:
                        esc = (
                            v.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                            .replace('"', "&quot;")
                        )
                        cells.append(
                            f'<c r="{r}" t="inlineStr"><is><t>{esc}</t></is></c>',
                        )
                row_xml.append(f"<row r=\"{r_ix+1}\">{''.join(cells)}</row>")
            return (
                f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<worksheet xmlns="{ns_main}" xmlns:r="{ns_rel}">'
                f"<sheetData>{''.join(row_xml)}</sheetData>"
                f"</worksheet>"
            )

        workbook_xml = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            f'<workbook xmlns="{ns_main}" xmlns:r="{ns_rel}">',
            "<sheets>",
        ]
        wb_rels = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        ]

        for i, (sheet_name, _rows) in enumerate(sheets, start=1):
            workbook_xml.append(
                f'<sheet name="{sheet_name}" sheetId="{i}" r:id="rId{i}"/>',
            )
            wb_rels.append(
                f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>',
            )

        # styles relation
        wb_rels.append(
            '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
        )
        workbook_xml.append("</sheets></workbook>")
        wb_rels.append("</Relationships>")

        content_types = dedent(
            """\
            <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
              <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
              <Default Extension="xml" ContentType="application/xml"/>
              <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
              <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
            </Types>
            """,
        ).strip()
        for i in range(1, len(sheets) + 1):
            content_types = content_types.replace(
                "</Types>",
                f'  <Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>\n</Types>',
            )

        rels_root = dedent(
            """\
            <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
              <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
            </Relationships>
            """,
        ).strip()

        styles_xml = dedent(
            f"""\
            <?xml version="1.0" encoding="UTF-8" standalone="yes"?>
            <styleSheet xmlns="{ns_main}">
              <fonts count="1"><font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font></fonts>
              <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
              <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
              <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
              <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
            </styleSheet>
            """,
        ).strip()

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", content_types)
            z.writestr("_rels/.rels", rels_root)
            z.writestr("xl/workbook.xml", "\n".join(workbook_xml))
            z.writestr("xl/_rels/workbook.xml.rels", "\n".join(wb_rels))
            z.writestr("xl/styles.xml", styles_xml)
            for i, (_sheet_name, rows) in enumerate(sheets, start=1):
                z.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(rows))

    return _write_minimal_xlsx


# ---------------------------------------------------------------------------
# ParseConfig hot-swap stubs (exposed as an importable module name)
# ---------------------------------------------------------------------------


class _BaseStubBackend(BaseFileParserBackend):
    """Shared helpers for test-only stub backends."""

    name: str = "stub_backend"
    supported_formats: Sequence[FileFormat] = ()

    def can_handle(self, fmt: Optional[FileFormat]) -> bool:
        return fmt in self.supported_formats

    def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
        # Provide non-empty summary + metadata so FileParser doesn't need to synthesize anything
        # and tests remain deterministic / LLM-free.
        return FileParseResult(
            logical_path=str(ctx.logical_path),
            status="success",
            file_format=ctx.file_format,
            mime_type=ctx.mime_type,
            summary=f"{self.name} summary",
            full_text="",
            metadata=FileParseMetadata(
                key_topics="stub_topic_1, stub_topic_2, stub_topic_3",
                named_entities="",
                content_tags="stub_tag_1, stub_tag_2, stub_tag_3, stub_tag_4, stub_tag_5",
                confidence_score=0.9,
            ),
            trace=None,  # Let FileParser attach trace
            graph=None,
            tables=[],
        )


class StubTxtBackendA(_BaseStubBackend):
    name = "stub_txt_a"
    supported_formats = (FileFormat.TXT,)


class StubTxtBackendB(_BaseStubBackend):
    name = "stub_txt_b"
    supported_formats = (FileFormat.TXT,)


class StubPdfBackend(_BaseStubBackend):
    name = "stub_pdf"
    supported_formats = (FileFormat.PDF,)

    def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
        out = super().parse(ctx)
        out.mime_type = out.mime_type or MimeType.APPLICATION_PDF
        return out


class StubCsvBackend(_BaseStubBackend):
    name = "stub_csv"
    supported_formats = (FileFormat.CSV,)

    def parse(self, ctx: FileParseRequest, /) -> FileParseResult:
        out = super().parse(ctx)
        out.mime_type = out.mime_type or MimeType.TEXT_CSV
        return out
