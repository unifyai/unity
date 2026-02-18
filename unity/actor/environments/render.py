"""Render environment for CodeActActor.

Provides a ``render`` namespace in the sandbox with methods for rendering
Excel sheets and PDF pages as PIL Images.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from unity.actor.environments.base import BaseEnvironment, ToolMetadata
from unity.function_manager.primitives.registry import get_registry

if TYPE_CHECKING:
    from PIL import Image
    from openpyxl.worksheet.worksheet import Worksheet


class _Renderer:
    """Runtime object injected into the sandbox as ``render``.

    Thin wrapper that delegates to the functions in
    ``unity.actor.environments.rendering``.
    """

    _PRIMITIVE_METHODS = ("render_excel_sheet", "render_pdf")

    @staticmethod
    def render_excel_sheet(
        sheet: "Worksheet",
        cell_range: str | None = None,
        scale: float = 1.0,
    ) -> "Image.Image":
        """Render an Excel sheet range as a PIL Image.

        Args:
            sheet: openpyxl Worksheet instance to render.
            cell_range: Range like ``"A1:J20"`` (default: full used range
                including charts/images).
            scale: Scale factor for output.

        Returns:
            PIL Image of the rendered range.
        """
        from unity.actor.environments.rendering import render_excel_sheet

        return render_excel_sheet(sheet, cell_range=cell_range, scale=scale)

    @staticmethod
    def render_pdf(
        source: "str | Path",
        page: int = 0,
        dpi: int = 150,
    ) -> "Image.Image":
        """Render a PDF page as a PIL Image.

        Args:
            source: Path to PDF file or pymupdf Document object.
            page: Page number to render (0-indexed, default: 0).
            dpi: Resolution for rendering (default: 150).

        Returns:
            PIL Image of the rendered PDF page.
        """
        from unity.actor.environments.rendering import render_pdf

        return render_pdf(source, page=page, dpi=dpi)


class RenderEnvironment(BaseEnvironment):
    """Environment that provides document rendering via the ``render`` namespace.

    Injects a ``render`` object into the sandbox with
    ``render_excel_sheet()`` and ``render_pdf()`` methods.
    """

    NAMESPACE = "render"

    def __init__(self) -> None:
        renderer = _Renderer()
        super().__init__(
            instance=renderer,
            namespace=self.NAMESPACE,
        )

    def get_tools(self) -> Dict[str, ToolMetadata]:
        return {
            f"{self.NAMESPACE}.render_excel_sheet": ToolMetadata(
                name=f"{self.NAMESPACE}.render_excel_sheet",
                is_impure=False,
                is_steerable=False,
            ),
            f"{self.NAMESPACE}.render_pdf": ToolMetadata(
                name=f"{self.NAMESPACE}.render_pdf",
                is_impure=False,
                is_steerable=False,
            ),
        }

    def get_prompt_context(self) -> str:
        """Generate prompt context from the rendering methods' docstrings."""
        registry = get_registry()
        lines = [f"### `{self.NAMESPACE}` — Document Rendering\n"]

        for method_name in _Renderer._PRIMITIVE_METHODS:
            method = getattr(_Renderer, method_name)
            sig_str = registry._format_method_signature(_Renderer, method_name)
            full_doc = inspect.getdoc(method) or ""
            filtered_doc = registry._filter_internal_params_from_docstring(full_doc)

            lines.append(f"**`{self.NAMESPACE}.{method_name}{sig_str}`**")
            if filtered_doc:
                for doc_line in filtered_doc.splitlines():
                    lines.append(f"  {doc_line}")
            lines.append("")

        return "\n".join(lines)

    async def capture_state(self) -> Dict[str, Any]:
        return {"type": "render"}
