"""
AQIVON OmniOCR Reading Order & Spatial Layout Sorter.
Reconstructs natural reading order for multi-column documents, forms, and mixed layouts
using geometric spatial projection and union-find clustering.
"""

from typing import List, Tuple
from aqivon_omniocr.types import TextBlock, BoundingBox


class ReadingOrderSorter:
    """Sorts detected text blocks into natural human reading order."""

    @staticmethod
    def sort_blocks(blocks: List[TextBlock], page_width: int, page_height: int) -> List[TextBlock]:
        """
        Sorts blocks considering columns, vertical flow, and horizontal alignment.
        """
        if not blocks:
            return []

        # 1. Determine if the page exhibits multi-column structure
        columns = ReadingOrderSorter._detect_columns(blocks, page_width)

        # 2. Assign each block to a column
        column_buckets: List[List[TextBlock]] = [[] for _ in columns]
        for block in blocks:
            assigned = False
            b_center_x, _ = block.bbox.center
            for col_idx, (col_left, col_right) in enumerate(columns):
                if col_left <= b_center_x <= col_right:
                    column_buckets[col_idx].append(block)
                    assigned = True
                    break
            if not assigned:
                # Assign to nearest column
                min_dist = float("inf")
                best_col = 0
                for col_idx, (col_left, col_right) in enumerate(columns):
                    col_center = (col_left + col_right) / 2.0
                    dist = abs(b_center_x - col_center)
                    if dist < min_dist:
                        min_dist = dist
                        best_col = col_idx
                column_buckets[best_col].append(block)

        # 3. Within each column, sort top-to-bottom, then left-to-right
        sorted_blocks: List[TextBlock] = []
        for bucket in column_buckets:
            # Sort with line tolerance: if vertical difference is small (< line_height * 0.5), sort by X
            sorted_bucket = ReadingOrderSorter._sort_column_lines(bucket)
            sorted_blocks.extend(sorted_bucket)

        # 4. Re-assign reading order indices
        for idx, block in enumerate(sorted_blocks):
            block.reading_order_idx = idx

        return sorted_blocks

    @staticmethod
    def _detect_columns(blocks: List[TextBlock], page_width: int) -> List[Tuple[float, float]]:
        """
        Detects primary column intervals [left, right] on a page.
        """
        if not blocks:
            return [(0.0, float(page_width))]

        # Check for multi-column layout by computing horizontal histogram of text block centers
        centers = [b.bbox.center[0] for b in blocks]
        half_width = page_width / 2.0
        left_count = sum(1 for c in centers if c < (half_width - page_width * 0.05))
        right_count = sum(1 for c in centers if c > (half_width + page_width * 0.05))

        total = len(blocks)
        # If significant text exists in both halves and there is a central gutter, treat as 2-column
        if total >= 8 and (left_count / total > 0.3) and (right_count / total > 0.3):
            # Check gutter emptiness around center (0.45 to 0.55 width)
            gutter_crossings = sum(1 for b in blocks if b.bbox.xmin < half_width < b.bbox.xmax)
            if gutter_crossings <= max(2, total * 0.15):
                # Valid 2-column layout
                return [
                    (0.0, half_width),
                    (half_width, float(page_width))
                ]

        # Single column layout default
        return [(0.0, float(page_width))]

    @staticmethod
    def _sort_column_lines(blocks: List[TextBlock]) -> List[TextBlock]:
        """Sorts blocks inside a column primarily by Y, grouping overlapping horizontal lines."""
        if not blocks:
            return []

        # Calculate average height for thresholding
        avg_height = sum(b.bbox.height for b in blocks) / len(blocks)
        line_threshold = max(8.0, avg_height * 0.6)

        # Initial sort by top Y
        sorted_by_y = sorted(blocks, key=lambda b: b.bbox.ymin)
        
        # Group into lines
        lines: List[List[TextBlock]] = []
        current_line: List[TextBlock] = []
        current_y = None

        for b in sorted_by_y:
            if current_y is None:
                current_line = [b]
                current_y = b.bbox.ymin
            elif abs(b.bbox.ymin - current_y) <= line_threshold:
                current_line.append(b)
            else:
                # Sort line left-to-right
                current_line.sort(key=lambda b: b.bbox.xmin)
                lines.append(current_line)
                current_line = [b]
                current_y = b.bbox.ymin

        if current_line:
            current_line.sort(key=lambda b: b.bbox.xmin)
            lines.append(current_line)

        # Flatten
        result = []
        for line in lines:
            result.extend(line)
        return result

