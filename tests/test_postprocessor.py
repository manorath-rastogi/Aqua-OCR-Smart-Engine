"""
Unit tests for AQIVON OmniOCR post-processor:
table reconstruction, reading order sorting, and markdown building.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aqivon_omniocr.types import BoundingBox, TextBlock, BlockType
from aqivon_omniocr.postprocessor import TableBuilder, ReadingOrderSorter, MarkdownBuilder


def test_table_html_parsing():
    html_sample = """
    <table>
        <tr><th>Item</th><th>Quantity</th><th>Price</th></tr>
        <tr><td>Product Alpha</td><td>2</td><td>$40</td></tr>
        <tr><td>Product Beta</td><td>1</td><td>$25</td></tr>
    </table>
    """
    res = TableBuilder.html_to_table_result(html_sample)
    assert res.num_rows == 3
    assert res.num_cols == 3
    assert len(res.cells) == 9
    assert "| Item | Quantity | Price |" in res.markdown
    assert "| Product Alpha | 2 | $40 |" in res.markdown


def test_reading_order_two_columns():
    page_width = 800
    page_height = 1000

    # Column 1 blocks (left half, x around 100)
    b1 = TextBlock("Left Column Line 1", BoundingBox(50, 100, 300, 130))
    b2 = TextBlock("Left Column Line 2", BoundingBox(50, 140, 300, 170))
    b3 = TextBlock("Left Column Line 3", BoundingBox(50, 180, 300, 210))
    b4 = TextBlock("Left Column Line 4", BoundingBox(50, 220, 300, 250))

    # Column 2 blocks (right half, x around 500)
    b5 = TextBlock("Right Column Line 1", BoundingBox(450, 100, 700, 130))
    b6 = TextBlock("Right Column Line 2", BoundingBox(450, 140, 700, 170))
    b7 = TextBlock("Right Column Line 3", BoundingBox(450, 180, 700, 210))
    b8 = TextBlock("Right Column Line 4", BoundingBox(450, 220, 700, 250))

    # Unsorted input list (interleaved)
    unsorted = [b5, b1, b6, b2, b7, b3, b8, b4]
    sorted_blocks = ReadingOrderSorter.sort_blocks(unsorted, page_width, page_height)

    # In 2-column layout, left column lines should come before right column lines
    texts = [b.text for b in sorted_blocks]
    assert texts[:4] == [
        "Left Column Line 1",
        "Left Column Line 2",
        "Left Column Line 3",
        "Left Column Line 4"
    ]
    assert texts[4:] == [
        "Right Column Line 1",
        "Right Column Line 2",
        "Right Column Line 3",
        "Right Column Line 4"
    ]


def test_clean_unlimited_ocr_tags():
    raw_vlm_output = """
    <PAGE_1>
    <title_1>AQIVON Technologies Executive Summary</title_1>
    This is an introductory paragraph describing OmniOCR.
    ![](page1_10_20_300_400.jpg)
    <title_2>System Performance</title_2>
    <table>
        <tr><th>Metric</th><th>Score</th></tr>
        <tr><td>Accuracy</td><td>99.4%</td></tr>
    </table>
    </PAGE_1>
    """
    clean_md = MarkdownBuilder.clean_raw_unlimited_ocr_output(raw_vlm_output)
    assert "**[Page 1]**" in clean_md
    assert "# AQIVON Technologies Executive Summary" in clean_md
    assert "## System Performance" in clean_md
    assert "| Metric | Score |" in clean_md
    assert "| Accuracy | 99.4% |" in clean_md
    assert "> *[Figure:" in clean_md


if __name__ == "__main__":
    test_table_html_parsing()
    test_reading_order_two_columns()
    test_clean_unlimited_ocr_tags()
    print("All post-processor tests passed successfully!")

