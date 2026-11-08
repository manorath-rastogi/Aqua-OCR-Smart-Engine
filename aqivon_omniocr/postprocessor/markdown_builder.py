"""
AQIVON OmniOCR Markdown & Layout Reconstruction Engine.
Adapted from Baidu Unlimited-OCR long-horizon post-processing and layout structuring.
Converts raw model tokens, tags, coordinates, and bounding boxes into clean, well-structured Markdown.
"""

import re
from typing import List, Optional, Tuple

from aqivon_omniocr.types import BoundingBox, TextBlock, BlockType, PageResult, TableResult
from aqivon_omniocr.postprocessor.table_builder import TableBuilder
from aqivon_omniocr.postprocessor.reading_order import ReadingOrderSorter


class MarkdownBuilder:
    """
    Transforms OCR text blocks, layout structures, and raw model output tokens
    into standardized, clean Markdown.
    """

    @staticmethod
    def blocks_to_markdown(blocks: List[TextBlock], tables: Optional[List[TableResult]] = None) -> str:
        """Constructs markdown from sorted text blocks and embedded tables."""
        if not blocks and not tables:
            return ""

        output_lines: List[str] = []
        
        for block in blocks:
            text = block.text.strip()
            if not text:
                continue

            # Format based on semantic block type
            if block.block_type == BlockType.HEADING_1:
                output_lines.append(f"\n# {text}\n")
            elif block.block_type == BlockType.HEADING_2:
                output_lines.append(f"\n## {text}\n")
            elif block.block_type == BlockType.HEADING_3:
                output_lines.append(f"\n### {text}\n")
            elif block.block_type == BlockType.LIST_ITEM:
                output_lines.append(f"- {text}")
            elif block.block_type == BlockType.FORMULA:
                if not text.startswith("$"):
                    text = f"$${text}$$"
                output_lines.append(f"\n{text}\n")
            elif block.block_type == BlockType.HANDWRITTEN:
                output_lines.append(f"*{text}*")  # Italicize handwritten segments
            else:
                output_lines.append(text)

        # Append structured tables if present
        if tables:
            for table in tables:
                if table.markdown:
                    output_lines.append("\n" + table.markdown.strip() + "\n")
                elif table.html:
                    output_lines.append("\n" + table.html.strip() + "\n")

        # Join and normalize multiple blank lines
        md_text = "\n".join(output_lines)
        md_text = re.sub(r"\n{3,}", "\n\n", md_text).strip()
        return md_text

    @staticmethod
    def clean_raw_unlimited_ocr_output(raw_output: str) -> str:
        """
        Cleans raw generation tokens emitted by Baidu Unlimited-OCR / VLM models.
        Normalizes page tags, heading levels, image coordinate placeholders, and table formatting.
        """
        text = raw_output

        # 1. Normalize page tags
        # Replace <PAGE_(\d+)> or <PAGE> with clean markdown page separators
        text = re.sub(r"<PAGE[_\s]*(\d+)?>", r"\n\n---\n**[Page \1]**\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</PAGE>", "\n", text, flags=re.IGNORECASE)

        # 2. Normalize title / heading tags
        # title<LEVEL1> -> # , title<LEVEL2> -> ##, etc.
        text = re.sub(r"(?:<title_?1>|title<LEVEL1>)(.*?)(?:</title_?1>|$)", r"\n# \1\n", text, flags=re.IGNORECASE)
        text = re.sub(r"(?:<title_?2>|title<LEVEL2>)(.*?)(?:</title_?2>|$)", r"\n## \1\n", text, flags=re.IGNORECASE)
        text = re.sub(r"(?:<title_?3>|title<LEVEL3>)(.*?)(?:</title_?3>|$)", r"\n### \1\n", text, flags=re.IGNORECASE)
        text = re.sub(r"(?:<title_?4>|title<LEVEL4>)(.*?)(?:</title_?4>|$)", r"\n#### \1\n", text, flags=re.IGNORECASE)

        # 3. Normalize image / figure coordinate tags
        # Format: ![](page<N>_x1_y1_x2_y2.jpg) or <figure coordinates="...">
        def _format_figure(match):
            coords = match.group(1) if match.lastindex >= 1 else ""
            return f"\n\n> *[Figure: {coords.strip()}]*\n\n"

        text = re.sub(r"!\[.*?\]\((page\d+_[0-9_]+\.(?:jpg|png))\)", _format_figure, text)
        text = re.sub(r"<figure[^>]*>(.*?)</figure>", _format_figure, text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<seal[^>]*>(.*?)</seal>", r"\n> *[Seal/Stamp: \1]*\n", text, flags=re.IGNORECASE)

        # 4. Clean and normalize embedded HTML tables
        def _clean_table_match(m):
            table_raw = m.group(0)
            table_res = TableBuilder.html_to_table_result(table_raw)
            if table_res.markdown:
                return f"\n\n{table_res.markdown}\n\n"
            return f"\n\n{table_res.html}\n\n"

        text = re.sub(r"<table.*?>.*?</table>", _clean_table_match, text, flags=re.DOTALL | re.IGNORECASE)

        # 5. Clean orphan special tokens
        text = re.sub(r"<(?:BLOCK|FORK|BOX|DET|SEP)[^>]*>", "", text)
        text = re.sub(r"</(?:BLOCK|FORK|BOX|DET|SEP)>", "", text)
        
        # 6. Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def extract_bboxes_from_unlimited_tags(text: str) -> List[Tuple[str, BoundingBox]]:
        """
        Extracts embedded bounding box labels from tags such as:
        <box>[x1, y1, x2, y2]</box> Content
        """
        extracted = []
        pattern = re.compile(r"<box>\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]</box>(.*?)(?=<box>|$)", re.DOTALL)
        for match in pattern.finditer(text):
            x1, y1, x2, y2 = map(float, match.groups()[:4])
            content = match.group(5).strip()
            bbox = BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2)
            extracted.append((content, bbox))
        return extracted

