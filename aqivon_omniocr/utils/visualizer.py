"""
AQIVON OmniOCR Visualizer Utility.
Draws color-coded bounding boxes, reading order indices, and layout blocks
onto document images for visual inspection and verification.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from aqivon_omniocr.types import BoundingBox, TextBlock, BlockType, PageResult


class LayoutVisualizer:
    """Visualizes detected bounding boxes, reading order, and block types on an image."""

    # Color scheme for different block types (BGR format for OpenCV)
    COLORS = {
        BlockType.HEADING_1: (180, 0, 180),    # Purple
        BlockType.HEADING_2: (200, 50, 150),   # Light Purple
        BlockType.HEADING_3: (160, 80, 200),   # Magenta
        BlockType.PARAGRAPH: (220, 100, 20),   # Blue
        BlockType.HANDWRITTEN: (30, 140, 255), # Orange
        BlockType.FORMULA: (0, 180, 255),      # Yellow
        BlockType.TABLE_CELL: (40, 200, 40),   # Green
        BlockType.LIST_ITEM: (200, 180, 0),    # Cyan
        BlockType.UNKNOWN: (120, 120, 120)     # Gray
    }

    @staticmethod
    def draw_page_result(
        image: np.ndarray,
        page_result: PageResult,
        show_reading_order: bool = True,
        show_confidence: bool = False
    ) -> np.ndarray:
        """Draws bounding boxes and labels onto RGB image, returning annotated RGB image."""
        annotated = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for block in page_result.blocks:
            bbox = block.bbox
            color = LayoutVisualizer.COLORS.get(block.block_type, (220, 100, 20))

            # Draw polygon if available, else rectangle
            if bbox.polygon and len(bbox.polygon) >= 4:
                pts = np.array([[int(p[0]), int(p[1])] for p in bbox.polygon], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
            else:
                x1, y1 = int(bbox.xmin), int(bbox.ymin)
                x2, y2 = int(bbox.xmax), int(bbox.ymax)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label tag
            label_parts = []
            if show_reading_order and block.reading_order_idx >= 0:
                label_parts.append(f"#{block.reading_order_idx + 1}")
            if block.is_handwritten:
                label_parts.append("[HW]")
            if show_confidence:
                label_parts.append(f"{int(block.confidence * 100)}%")

            if label_parts:
                label_str = " ".join(label_parts)
                x_tag = int(bbox.xmin)
                y_tag = max(15, int(bbox.ymin) - 5)
                # Background pill for readability
                (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x_tag, y_tag - th - 2), (x_tag + tw + 4, y_tag + 2), color, -1)
                cv2.putText(annotated, label_str, (x_tag + 2, y_tag - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

