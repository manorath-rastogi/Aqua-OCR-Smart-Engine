"""
Aqua OCR Smart Engine Advanced Reading Order & Multi-Column Spatial Layout Sorter.
Reconstructs natural human reading order for complex documents:
- Multi-column resumes, academic papers, and newsletters
- Form headers, banners, and footnotes
- Semantic block categorization (Headings, Paragraphs, List Items, Captions)
"""

import re
from typing import List, Tuple
import numpy as np

from aqua_ocr_smart_engine.types import TextBlock, BlockType, BoundingBox


class ReadingOrderSorter:
    """Sorts detected text blocks into natural human reading order preserving layout columns."""

    BULLET_PATTERN = re.compile(r"^(\s*[\u2022\u2023\u25E6\u2043\u2219\u00B7\*\-\–\—\•▪►]\s*|\s*\d+[\.\)]\s*|\s*[a-zA-Z][\.\)]\s+)")

    @staticmethod
    def sort_blocks(blocks: List[TextBlock], page_width: int, page_height: int) -> List[TextBlock]:
        """
        Sorts blocks considering natural multi-column flow, headers, and footers.
        """
        if not blocks:
            return []

        # 1. Header and Footer separation
        header_y_limit = page_height * 0.12
        footer_y_limit = page_height * 0.94

        header_candidates: List[TextBlock] = []
        footer_candidates: List[TextBlock] = []
        body_candidates: List[TextBlock] = []

        for b in blocks:
            # Header: top 12% of page
            if b.bbox.ymin <= header_y_limit:
                header_candidates.append(b)
            # Footer: bottom 6% of page
            elif b.bbox.ymin >= footer_y_limit:
                footer_candidates.append(b)
            else:
                body_candidates.append(b)

        # 2. Multi-column detection on body blocks
        columns = ReadingOrderSorter._detect_body_columns(body_candidates, page_width)

        # 3. Assign body blocks to columns
        column_buckets: List[List[TextBlock]] = [[] for _ in columns]
        for b in body_candidates:
            b_center_x = b.bbox.center[0]
            assigned = False
            for col_idx, (col_left, col_right) in enumerate(columns):
                if col_left <= b_center_x <= col_right:
                    column_buckets[col_idx].append(b)
                    b.column_id = col_idx + 1
                    assigned = True
                    break
            if not assigned:
                # Nearest column fallback
                dists = [abs(b_center_x - (left + right) / 2.0) for left, right in columns]
                best_idx = int(np.argmin(dists))
                column_buckets[best_idx].append(b)
                b.column_id = best_idx + 1

        # 4. Sort Header blocks (full width, reading order)
        for b in header_candidates:
            b.column_id = 0
        sorted_header = ReadingOrderSorter._sort_column_lines(header_candidates)

        # 5. Sort each Column independently top-to-bottom
        sorted_body: List[TextBlock] = []
        for bucket in column_buckets:
            sorted_bucket = ReadingOrderSorter._sort_column_lines(bucket)
            sorted_body.extend(sorted_bucket)

        # 6. Sort Footer blocks
        for b in footer_candidates:
            b.column_id = 0
        sorted_footer = ReadingOrderSorter._sort_column_lines(footer_candidates)

        # Combine into complete reading order
        full_sorted = sorted_header + sorted_body + sorted_footer

        # 7. Semantic enhancement: Detect headings, bullet points, and assign sequential indices
        for idx, block in enumerate(full_sorted):
            block.reading_order_idx = idx
            ReadingOrderSorter._enrich_block_semantics(block, page_width)

        return full_sorted

    @staticmethod
    def _detect_body_columns(blocks: List[TextBlock], page_width: int) -> List[Tuple[float, float]]:
        """
        Detects body columns by analyzing the horizontal text occupancy profile.
        Finds true vertical whitespace gutters between columns.
        """
        if len(blocks) < 6:
            return [(0.0, float(page_width))]

        profile = np.zeros(page_width, dtype=int)
        for b in blocks:
            x1 = max(0, min(page_width - 1, int(b.bbox.xmin)))
            x2 = max(0, min(page_width, int(b.bbox.xmax)))
            profile[x1:x2] += 1

        # Search for gutters in the central 30% - 75% region of the page
        search_start = int(page_width * 0.28)
        search_end = int(page_width * 0.78)
        search_region = profile[search_start:search_end]

        # A gutter is a continuous span of zero or minimal text occupancy
        min_gutter_width = max(10, int(page_width * 0.015))
        threshold = max(0, int(np.percentile(search_region, 10)))

        valley_mask = (search_region <= threshold)
        valleys = np.where(valley_mask)[0] + search_start

        if len(valleys) == 0:
            return [(0.0, float(page_width))]

        # Find continuous valley segments
        gaps: List[Tuple[int, int, int]] = []
        gap_start = valleys[0]
        prev = gap_start
        for v in valleys[1:]:
            if v == prev + 1:
                prev = v
            else:
                width = prev - gap_start + 1
                if width >= min_gutter_width:
                    gaps.append((gap_start, prev, width))
                gap_start = v
                prev = v
        width = prev - gap_start + 1
        if width >= min_gutter_width:
            gaps.append((gap_start, prev, width))

        if not gaps:
            return [(0.0, float(page_width))]

        # Pick the most prominent gutter
        gaps.sort(key=lambda g: g[2], reverse=True)
        best_gap = gaps[0]
        gutter_center = (best_gap[0] + best_gap[1]) / 2.0

        # Verify that both sides of the gutter have substantial content (at least 20% each)
        left_count = sum(1 for b in blocks if b.bbox.center[0] < gutter_center)
        right_count = sum(1 for b in blocks if b.bbox.center[0] > gutter_center)

        if left_count >= len(blocks) * 0.20 and right_count >= len(blocks) * 0.20:
            # Valid 2-column layout
            return [
                (0.0, gutter_center),
                (gutter_center, float(page_width))
            ]

        return [(0.0, float(page_width))]

    @staticmethod
    def _sort_column_lines(blocks: List[TextBlock]) -> List[TextBlock]:
        """Sorts blocks inside a column primarily by Y, grouping overlapping horizontal lines."""
        if not blocks:
            return []

        avg_height = sum(b.bbox.height for b in blocks) / len(blocks)
        line_threshold = max(6.0, avg_height * 0.55)

        # Initial sort by top Y
        sorted_by_y = sorted(blocks, key=lambda b: b.bbox.ymin)

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

        # Flatten lines in top-to-bottom reading order
        result = []
        for line in lines:
            result.extend(line)
        return result

    @staticmethod
    def _enrich_block_semantics(block: TextBlock, page_width: int):
        """Enriches block classification with headings, list items, and subtitles."""
        text = block.text.strip()
        if not text:
            return

        # 1. Bullet point / List item detection
        if ReadingOrderSorter.BULLET_PATTERN.match(text):
            block.block_type = BlockType.LIST_ITEM
            return

        # 2. Heading detection:
        # - Short text (< 45 chars)
        # - Uppercase or title case
        # - High relative font size or prominent section headers
        is_uppercase = text.isupper() and len(text) > 2
        is_known_heading = text.lower() in {
            "summary", "experience", "education", "skills", "projects",
            "key achievements", "certifications", "training & courses",
            "interests", "languages", "awards", "publications", "objective",
            "work experience", "professional experience", "invoice",
            "bill to", "ship to", "item description", "service summary"
        }

        if is_known_heading or (is_uppercase and len(text) <= 40):
            block.block_type = BlockType.HEADING_2
        elif block.reading_order_idx == 0 and len(text) <= 35:
            # Candidate or document title
            block.block_type = BlockType.HEADING_1
