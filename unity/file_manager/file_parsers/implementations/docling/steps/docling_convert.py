"""
Docling conversion helpers.

This is a thin wrapper around Docling's `DocumentConverter` with:
- typed outputs
- structured errors
- centralized settings usage (no ad-hoc env var reads)

We keep the surface small so it can be swapped out later (e.g., replace Docling
for certain formats like XLSX).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from unity.file_manager.file_parsers.settings import FileParserSettings
from unity.file_manager.file_parsers.types.contracts import ParseError


@dataclass(frozen=True)
class DoclingConvertResult:
    ok: bool
    document: Optional[object]
    error: Optional[ParseError] = None


def new_docling_converter(*, settings: FileParserSettings):
    """
    Construct a Docling `DocumentConverter`.

    Notes
    -----
    - We configure the PDF pipeline to enable picture extraction and (optionally)
      picture description. This mirrors the previous monolithic parser defaults.
    """
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except Exception as e:
        raise RuntimeError(
            "Docling is required for Docling-backed parsing but is not available",
        ) from e

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True

    if settings.PICTURE_DESCRIPTION_ENABLED:
        from docling.datamodel.pipeline_options import (
            PictureDescriptionVlmOptions,
        )
        from unity.file_manager.file_parsers.prompts.image_prompts import (
            build_picture_description_prompt,
        )

        pipeline_options.do_picture_description = True
        pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
            repo_id=settings.PICTURE_DESCRIPTION_MODEL_REPO,
            prompt=build_picture_description_prompt(),
        )

    return DocumentConverter(
        allowed_formats=list(InputFormat),
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def docling_convert(*, converter, source: str) -> DoclingConvertResult:
    """Convert a source path into a DoclingDocument."""
    try:
        from docling.datamodel.base_models import ConversionStatus
    except Exception as e:
        return DoclingConvertResult(
            ok=False,
            document=None,
            error=ParseError(
                code="docling_import_error",
                message="Docling ConversionStatus import failed",
                exception_type=type(e).__name__,
                details={"source": source},
            ),
        )

    try:
        res = converter.convert(source=source)
    except Exception as e:
        return DoclingConvertResult(
            ok=False,
            document=None,
            error=ParseError(
                code="docling_convert_exception",
                message=str(e),
                exception_type=type(e).__name__,
                details={"source": source},
            ),
        )

    if getattr(res, "status", None) != ConversionStatus.SUCCESS:
        return DoclingConvertResult(
            ok=False,
            document=None,
            error=ParseError(
                code="docling_convert_failed",
                message=f"Docling conversion failed with status: {getattr(res, 'status', None)}",
                exception_type=None,
                details={"source": source},
            ),
        )

    return DoclingConvertResult(ok=True, document=getattr(res, "document", None))
