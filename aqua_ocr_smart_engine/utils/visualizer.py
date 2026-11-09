"""
Aqua OCR Smart Engine Visualizer Utility.
Draws color-coded bounding boxes, reading order badges, and layout blocks
onto document images for visual inspection and verification.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from aqua_ocr_smart_engine.types import BoundingBox, TextBlock, BlockType, PageResult


class LayoutVisualizer:
    """Visualizes detected bounding boxes, reading order, and block types on an image."""

    # Color scheme for different block types (RGB format for PIL / Matplotlib / Gradio)
    COLORS_RGB = {
        BlockType.HEADING_1: (156, 39, 176),   # Purple
        BlockType.HEADING_2: (186, 104, 200),  # Light Purple
        BlockType.HEADING_3: (142, 36, 170),   # Magenta
        BlockType.PARAGRAPH: (33, 150, 243),   # Material Blue
        BlockType.HANDWRITTEN: (255, 112, 67), # Deep Orange
        BlockType.FORMULA: (255, 179, 0),      # Amber/Yellow
        BlockType.TABLE_CELL: (76, 175, 80),   # Green
        BlockType.LIST_ITEM: (0, 188, 212),    # Cyan
        BlockType.UNKNOWN: (158, 158, 158)     # Grey
    }

    @staticmethod
    def draw_page_result(
        image: np.ndarray,
        page_result: PageResult,
        show_reading_order: bool = True,
        show_confidence: bool = False,
        min_confidence: float = 0.0
    ) -> np.ndarray:
        """
        Draws bounding boxes and badges onto RGB image, returning annotated RGB image.
        """
        # Work in BGR for OpenCV drawing
        annotated = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        h, w = annotated.shape[:2]

        for block in page_result.blocks:
            if block.confidence < min_confidence:
                continue

            bbox = block.bbox
            color_rgb = LayoutVisualizer.COLORS_RGB.get(block.block_type, (33, 150, 243))
            # Convert RGB to BGR for OpenCV
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            # Draw polygon if available, else rectangle
            if bbox.polygon and len(bbox.polygon) >= 4:
                pts = np.array([[int(p[0]), int(p[1])] for p in bbox.polygon], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=True, color=color_bgr, thickness=2)
            else:
                x1, y1 = max(0, int(bbox.xmin)), max(0, int(bbox.ymin))
                x2, y2 = min(w, int(bbox.xmax)), min(h, int(bbox.ymax))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

            # Build label tag
            label_parts = []
            if show_reading_order and block.reading_order_idx >= 0:
                label_parts.append(f"#{block.reading_order_idx + 1}")
            if block.is_handwritten or block.block_type == BlockType.HANDWRITTEN:
                label_parts.append("[HW]")
            if show_confidence:
                label_parts.append(f"{int(block.confidence * 100)}%")

            if label_parts:
                label_str = " ".join(label_parts)
                x_tag = max(0, int(bbox.xmin))
                y_tag = max(18, int(bbox.ymin) - 4)

                font_scale = 0.45
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Pill background
                cv2.rectangle(
                    annotated,
                    (x_tag, y_tag - th - baseline - 2),
                    (x_tag + tw + 6, y_tag + baseline),
                    color_bgr,
                    -1
                )
                # Text in white
                cv2.putText(
                    annotated,
                    label_str,
                    (x_tag + 3, y_tag - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
