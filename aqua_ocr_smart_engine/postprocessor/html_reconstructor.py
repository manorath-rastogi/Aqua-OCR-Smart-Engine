"""
Aqua OCR Smart Engine HTML Document Layout Reconstructor.
Reconstructs original visual document layouts (columns, headings, bullet lists, tables)
into structured, beautifully styled HTML that renders side-by-side with original scans.
"""

import html
from typing import List
from aqua_ocr_smart_engine.types import PageResult, TextBlock, BlockType


class HTMLDocumentReconstructor:
    """Renders structured document pages into layout-faithful HTML."""

    DOCUMENT_CSS = """
    <style>
      .aqua-doc-root {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        line-height: 1.5;
        color: #212529;
        background: #ffffff;
        border-radius: 8px;
        padding: 32px 36px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        max-width: 950px;
        margin: 0 auto;
        border: 1px solid #e9ecef;
      }
      .dark .aqua-doc-root {
        background: #1a1d20;
        color: #e9ecef;
        border-color: #2c3036;
      }
      .aqua-doc-header {
        border-bottom: 2px solid #0d6efd;
        padding-bottom: 16px;
        margin-bottom: 24px;
      }
      .aqua-doc-header h1 {
        margin: 0 0 6px 0;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0f172a;
      }
      .dark .aqua-doc-header h1 { color: #f8fafc; }
      .aqua-doc-header .aqua-subtitle {
        font-size: 15px;
        color: #64748b;
        margin: 0 0 8px 0;
        font-weight: 500;
      }
      .dark .aqua-doc-header .aqua-subtitle { color: #94a3b8; }
      .aqua-doc-columns {
        display: grid;
        gap: 32px;
        align-items: start;
      }
      .aqua-col-1 {
        min-width: 0;
      }
      .aqua-col-2 {
        min-width: 0;
        border-left: 1px dashed #e2e8f0;
        padding-left: 28px;
      }
      .dark .aqua-col-2 {
        border-left-color: #334155;
      }
      .aqua-section-heading {
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #0284c7;
        border-bottom: 1.5px solid #0284c7;
        padding-bottom: 4px;
        margin: 22px 0 12px 0;
      }
      .dark .aqua-section-heading { color: #38bdf8; border-bottom-color: #38bdf8; }
      .aqua-subheading {
        font-size: 15px;
        font-weight: 600;
        color: #1e293b;
        margin: 14px 0 4px 0;
      }
      .dark .aqua-subheading { color: #e2e8f0; }
      .aqua-p {
        font-size: 13.5px;
        color: #334155;
        margin: 0 0 10px 0;
        line-height: 1.6;
      }
      .dark .aqua-p { color: #cbd5e1; }
      .aqua-list {
        margin: 6px 0 14px 0;
        padding-left: 20px;
        list-style-type: disc;
      }
      .aqua-list li {
        font-size: 13px;
        color: #334155;
        margin-bottom: 6px;
        line-height: 1.55;
      }
      .dark .aqua-list li { color: #cbd5e1; }
      .aqua-doc-footer {
        margin-top: 32px;
        padding-top: 14px;
        border-top: 1px solid #e2e8f0;
        font-size: 12px;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
      }
      .aqua-badge-order {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        padding: 1px 5px;
        background: rgba(13, 110, 253, 0.1);
        color: #0d6efd;
        border-radius: 4px;
        margin-right: 6px;
        vertical-align: middle;
      }
    </style>
    """

    @classmethod
    def render_page(cls, page: PageResult) -> str:
        """Renders an individual PageResult into a structured HTML document."""
        if not page or not page.blocks:
            return "<div class='aqua-doc-root'><p style='color:#888;'>No content detected on page.</p></div>"

        blocks = page.blocks
        h = max(1, page.height)
        w = max(1, page.width)

        # Partition into sections based on column_id
        col_ids = set(b.column_id for b in blocks)
        is_multi_column = len(col_ids - {0}) >= 2

        header_blocks = [b for b in blocks if b.column_id == 0 and b.bbox.ymin < h * 0.18]
        footer_blocks = [b for b in blocks if b.column_id == 0 and b.bbox.ymin >= h * 0.85]

        # Body blocks
        body_blocks = [b for b in blocks if b not in header_blocks and b not in footer_blocks]

        html_parts = [cls.DOCUMENT_CSS, "<div class='aqua-doc-root'>"]

        # Render Header
        if header_blocks:
            html_parts.append("<div class='aqua-doc-header'>")
            # First block is usually title/name
            first = header_blocks[0]
            html_parts.append(f"<h1>{html.escape(first.text)}</h1>")
            for hb in header_blocks[1:]:
                html_parts.append(f"<div class='aqua-subtitle'>{html.escape(hb.text)}</div>")
            html_parts.append("</div>")

        # Render Multi-Column Body
        if is_multi_column:
            col1_blocks = [b for b in body_blocks if b.column_id == 1]
            col2_blocks = [b for b in body_blocks if b.column_id == 2]

            # Approximate width ratios
            col1_w = max(100.0, max((b.bbox.xmax for b in col1_blocks), default=w * 0.55))
            ratio1 = max(1.1, min(2.0, (col1_w / w) * 2.2))
            ratio2 = 1.0

            html_parts.append(f"<div class='aqua-doc-columns' style='grid-template-columns: {ratio1:.1f}fr {ratio2:.1f}fr;'>")
            
            # Left Column
            html_parts.append("<div class='aqua-col-1'>")
            html_parts.append(cls._render_block_stream(col1_blocks))
            html_parts.append("</div>")

            # Right Column
            html_parts.append("<div class='aqua-col-2'>")
            html_parts.append(cls._render_block_stream(col2_blocks))
            html_parts.append("</div>")

            html_parts.append("</div>")
        else:
            # Single Column Body
            html_parts.append("<div class='aqua-doc-body'>")
            html_parts.append(cls._render_block_stream(body_blocks))
            html_parts.append("</div>")

        # Render Footer
        if footer_blocks:
            html_parts.append("<div class='aqua-doc-footer'>")
            footer_text = " &bull; ".join(html.escape(b.text) for b in footer_blocks)
            html_parts.append(f"<span>{footer_text}</span>")
            html_parts.append(f"<span>Page {page.page_number}</span>")
            html_parts.append("</div>")

        html_parts.append("</div>")
        return "\n".join(html_parts)

    @classmethod
    def _render_block_stream(cls, blocks: List[TextBlock]) -> str:
        """Converts an ordered list of TextBlocks into semantic HTML elements."""
        out = []
        in_list = False

        for b in blocks:
            text = b.text.strip()
            if not text:
                continue

            escaped = html.escape(text)

            # Check for bullet point
            if b.block_type == BlockType.LIST_ITEM or text.startswith(("•", "-", "*", "▪", "·")):
                # Strip leading bullet char
                clean_item = text.lstrip("•-*▪· \t")
                if not in_list:
                    out.append("<ul class='aqua-list'>")
                    in_list = True
                out.append(f"<li>{html.escape(clean_item)}</li>")
                continue
            else:
                if in_list:
                    out.append("</ul>")
                    in_list = False

            if b.block_type in (BlockType.HEADING_1, BlockType.HEADING_2):
                out.append(f"<div class='aqua-section-heading'>{escaped}</div>")
            elif b.block_type == BlockType.HEADING_3:
                out.append(f"<div class='aqua-subheading'>{escaped}</div>")
            else:
                out.append(f"<p class='aqua-p'>{escaped}</p>")

        if in_list:
            out.append("</ul>")

        return "\n".join(out)

