"""
AQIVON OmniOCR Table Reconstruction & Formatting Module.
Reconstructs, sanitizes, and converts table structures from OCR tokens, HTML,
and grid cells into clean Markdown and semantic HTML representations.
"""

from html.parser import HTMLParser
import re
from typing import Any, Dict, List, Optional, Tuple

from aqivon_omniocr.types import BoundingBox, TableCell, TableResult


class TableBuilder:
    """Constructs and cleans tables for structured document output."""

    @staticmethod
    def cells_to_markdown(cells: List[TableCell], num_rows: int, num_cols: int) -> str:
        """Converts an array of TableCell objects into a standard Markdown table."""
        if num_rows == 0 or num_cols == 0:
            return ""

        # Create 2D grid
        grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
        for cell in cells:
            r = min(cell.row_idx, num_rows - 1)
            c = min(cell.col_idx, num_cols - 1)
            # Clean text: replace internal newlines and pipe characters
            clean_text = cell.text.replace("\n", " ").replace("|", "\\|").strip()
            grid[r][c] = clean_text

        # Format rows
        lines = []
        # Header row (row 0)
        header_row = "| " + " | ".join(grid[0]) + " |"
        separator = "| " + " | ".join(["---"] * num_cols) + " |"
        lines.append(header_row)
        lines.append(separator)

        for r in range(1, num_rows):
            row_str = "| " + " | ".join(grid[r]) + " |"
            lines.append(row_str)

        return "\n".join(lines)

    @staticmethod
    def html_to_table_result(html_str: str, bbox: Optional[BoundingBox] = None) -> TableResult:
        """Parses HTML table markup (e.g., from Baidu Unlimited-OCR / PP-Structure) into TableResult."""
        parser = _HTMLTableExtractor()
        parser.feed(html_str)
        
        cells = parser.cells
        num_rows = parser.max_row + 1 if parser.max_row >= 0 else 0
        num_cols = parser.max_col + 1 if parser.max_col >= 0 else 0
        
        # Format markdown
        md = TableBuilder.cells_to_markdown(cells, num_rows, num_cols)
        clean_html = TableBuilder.sanitize_table_html(html_str)

        return TableResult(
            bbox=bbox,
            num_rows=num_rows,
            num_cols=num_cols,
            cells=cells,
            html=clean_html,
            markdown=md
        )

    @staticmethod
    def sanitize_table_html(html_str: str) -> str:
        """Removes extraneous tags, normalizes colspan/rowspan attributes, and ensures well-formed HTML."""
        # Remove script and style tags
        cleaned = re.sub(r"<(script|style).*?</\1>", "", html_str, flags=re.DOTALL | re.IGNORECASE)
        # Normalize whitespace in tags
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Ensure <table> and </table> exist
        if not cleaned.lower().startswith("<table"):
            cleaned = f"<table>{cleaned}"
        if not cleaned.lower().endswith("</table>"):
            cleaned = f"{cleaned}</table>"
        return cleaned


class _HTMLTableExtractor(HTMLParser):
    """Internal HTML parser to extract cells, rows, and handle spans."""

    def __init__(self):
        super().__init__()
        self.cells: List[TableCell] = []
        self.current_row = -1
        self.current_col = -1
        self.in_cell = False
        self.cell_tag = ""
        self.cell_text_buffer: List[str] = []
        self.current_row_span = 1
        self.current_col_span = 1
        self.max_row = -1
        self.max_col = -1
        # Track occupied grid positions due to rowspan/colspan
        self.occupied: set[Tuple[int, int]] = set()

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        tag = tag.lower()
        if tag == "tr":
            self.current_row += 1
            self.current_col = -1
            if self.current_row > self.max_row:
                self.max_row = self.current_row
        elif tag in ("td", "th"):
            self.in_cell = True
            self.cell_tag = tag
            self.cell_text_buffer = []
            
            # Find next unoccupied column in this row
            self.current_col += 1
            while (self.current_row, self.current_col) in self.occupied:
                self.current_col += 1

            # Parse rowspan and colspan
            attr_dict = {k.lower(): (v or "") for k, v in attrs}
            try:
                self.current_row_span = max(1, int(attr_dict.get("rowspan", 1)))
            except ValueError:
                self.current_row_span = 1
            try:
                self.current_col_span = max(1, int(attr_dict.get("colspan", 1)))
            except ValueError:
                self.current_col_span = 1

            # Mark occupied
            for r in range(self.current_row, self.current_row + self.current_row_span):
                for c in range(self.current_col, self.current_col + self.current_col_span):
                    self.occupied.add((r, c))
                    if r > self.max_row:
                        self.max_row = r
                    if c > self.max_col:
                        self.max_col = c

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in ("td", "th") and self.in_cell:
            cell_text = "".join(self.cell_text_buffer).strip()
            cell = TableCell(
                row_idx=self.current_row,
                col_idx=self.current_col,
                row_span=self.current_row_span,
                col_span=self.current_col_span,
                text=cell_text,
                is_header=(self.cell_tag == "th")
            )
            self.cells.append(cell)
            self.in_cell = False
            self.cell_text_buffer = []

    def handle_data(self, data: str):
        if self.in_cell:
            self.cell_text_buffer.append(data)

