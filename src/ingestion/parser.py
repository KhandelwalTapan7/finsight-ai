"""
Multimodal PDF parsing.

Design choice: instead of a separate CLIP image-embedding pipeline, chart
and figure pages are rendered to an image and described by a free-tier
vision LLM (Groq). That description is embedded as ordinary text, in the
same vector space as everything else. This is simpler to get right than a
dual-embedding-space setup and is a well-established pattern for
multimodal RAG ("vision-to-text bridging").

Nothing here is domain-specific — the same pipeline handles a contract,
a research paper, a manual or an annual report.

Each page produces zero or more Chunks tagged with a source_type:
  - "text"  : the page's extracted text
  - "table" : a table pdfplumber found on the page (kept as markdown)
  - "chart" : a vision-model description of the rendered page image,
              only produced when the page looks image/chart heavy
              (little extractable text relative to its visual content)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import base64
import io

import fitz  # pymupdf
import pdfplumber
from groq import Groq

from src.config import settings

SourceType = Literal["text", "table", "chart"]


@dataclass
class RawChunk:
    doc_id: str
    page: int
    source_type: SourceType
    content: str
    meta: dict = field(default_factory=dict)


def _table_to_markdown(table: list[list[str | None]]) -> str:
    if not table or not table[0]:
        return ""
    header = table[0]
    rows = table[1:]
    md = "| " + " | ".join(str(c or "") for c in header) + " |\n"
    md += "|" + "|".join(["---"] * len(header)) + "|\n"
    for row in rows:
        md += "| " + " | ".join(str(c or "") for c in row) + " |\n"
    return md


def _page_looks_chart_heavy(page_text: str, image_count: int) -> bool:
    # Heuristic: lots of images/drawings but thin text -> likely a chart/
    # infographic page rather than a prose page. Good enough for a
    # portfolio project; a production system would tune this threshold
    # against labeled pages.
    return image_count > 0 and len(page_text.strip()) < 400


def _describe_chart_with_vision(png_bytes: bytes) -> str:
    if not settings.GROQ_API_KEY:
        return "[chart description skipped: GROQ_API_KEY not set]"
    client = Groq(api_key=settings.GROQ_API_KEY)
    b64 = base64.b64encode(png_bytes).decode()
    resp = client.chat.completions.create(
        model=settings.GROQ_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This image is a page from a document. Describe "
                            "every chart, graph, diagram, figure or infographic "
                            "on it precisely: its type, axis or node labels, "
                            "series or category names, and the key values, "
                            "relationships or trends it shows. If it is a "
                            "photograph, schematic or screenshot instead, "
                            "describe its content and any visible text. If "
                            "there is no visual element, say so briefly."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=500,
        temperature=0.1,
    )
    return resp.choices[0].message.content


def parse_pdf(file_path: str | Path, doc_id: str) -> list[RawChunk]:
    """Parse a single PDF into text / table / chart-description chunks."""
    file_path = Path(file_path)
    chunks: list[RawChunk] = []

    doc = fitz.open(file_path)
    with pdfplumber.open(file_path) as pdf_plumber_doc:
        for page_num in range(len(doc)):
            fitz_page = doc[page_num]
            page_text = fitz_page.get_text()
            image_count = len(fitz_page.get_images(full=True))

            if page_text.strip():
                chunks.append(
                    RawChunk(
                        doc_id=doc_id,
                        page=page_num + 1,
                        source_type="text",
                        content=page_text.strip(),
                    )
                )

            # Tables (pdfplumber is much better at table detection than pymupdf)
            plumber_page = pdf_plumber_doc.pages[page_num]
            for t_idx, table in enumerate(plumber_page.extract_tables()):
                md = _table_to_markdown(table)
                if md:
                    chunks.append(
                        RawChunk(
                            doc_id=doc_id,
                            page=page_num + 1,
                            source_type="table",
                            content=md,
                            meta={"table_index": t_idx},
                        )
                    )

            # Charts: render page -> vision model description
            if _page_looks_chart_heavy(page_text, image_count):
                pix = fitz_page.get_pixmap(dpi=150)
                png_bytes = pix.tobytes("png")
                description = _describe_chart_with_vision(png_bytes)
                chunks.append(
                    RawChunk(
                        doc_id=doc_id,
                        page=page_num + 1,
                        source_type="chart",
                        content=description,
                    )
                )

    doc.close()
    return chunks
