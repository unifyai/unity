from __future__ import annotations


def build_picture_description_prompt() -> str:
    """Prompt for picture/figure description retaining all factual details."""
    return (
        "Explain clearly what the image depicts and conveys — do not merely list items.\n"
        "- Overview: one or two sentences stating the figure's purpose and subject.\n"
        "- Detailed explanation (use exact labels/terms from the image):\n"
        "  • If a flowchart/block diagram: walk through the full process in order, naming each node/connector exactly and indicating direction/conditions.\n"
        "    Rather than repeating which node is connected to which, explain in depth the actual flow of the process(es) that is being depicted in the diagram.\n"
        "  • If a chart/graph: name axes/series, describe trends, peaks, changes, correlations, and comparative differences; provide values where readable.\n"
        "  • If a table: summarize headers and the most important rows/columns with exact figures.\n"
        "- Transcribe visible text verbatim (titles, labels, legends, annotations).\n"
        "- Retain ALL numeric values exactly as shown (units, scales, percentages, dates, counts).\n"
        "- Reason about relationships and implications that are evident from the image (sequence, cause–effect, comparisons), without guessing beyond what is visible.\n"
        "- Mention colors/shapes/symbols only when they encode meaning (e.g., legend mapping).\n"
        "- Use short paragraphs or bullets; remain strictly factual and non-speculative."
    )
