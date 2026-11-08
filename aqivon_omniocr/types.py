"""
AQIVON OmniOCR Standardized Types & Data Models.
Provides a unified representation of OCR detections, pages, tables, and documents.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Tuple, Union


class BlockType(str, Enum):
    """Semantic classification of an OCR text block."""
    PARAGRAPH = "paragraph"
    HEADING_1 = "heading_1"
    HEADING_2 = "heading_2"
    HEADING_3 = "heading_3"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    FORMULA = "formula"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    HANDWRITTEN = "handwritten"
    SIGNATURE = "signature"
    KEY_VALUE = "key_value"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box representation in 2D coordinates."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    polygon: Optional[List[Tuple[float, float]]] = None  # 4 points for rotated bboxes: [(x1,y1), (x2,y2), ...]

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_rect_list(self) -> List[float]:
        """Returns [xmin, ymin, xmax, ymax]."""
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "xmin": round(self.xmin, 2),
            "ymin": round(self.ymin, 2),
            "xmax": round(self.xmax, 2),
            "ymax": round(self.ymax, 2),
            "polygon": [[round(p[0], 2), round(p[1], 2)] for p in self.polygon] if self.polygon else None
        }

    @classmethod
    def from_polygon(cls, polygon: List[Union[List[float], Tuple[float, float]]]) -> "BoundingBox":
        """Calculates enclosing bbox from 4-point polygon."""
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        poly_tuples = [(float(p[0]), float(p[1])) for p in polygon]
        return cls(
            xmin=float(min(xs)),
            ymin=float(min(ys)),
            xmax=float(max(xs)),
            ymax=float(max(ys)),
            polygon=poly_tuples
        )


@dataclass
class TextBlock:
    """A recognized line or segment of text."""
    text: str
    bbox: BoundingBox
    confidence: float = 1.0
    language: Optional[str] = None
    block_type: BlockType = BlockType.PARAGRAPH
    is_handwritten: bool = False
    reading_order_idx: int = -1
    font_size_estimate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "confidence": round(self.confidence, 4),
            "language": self.language,
            "block_type": self.block_type.value,
            "is_handwritten": self.is_handwritten,
            "reading_order_idx": self.reading_order_idx,
            "font_size_estimate": self.font_size_estimate
        }


@dataclass
class TableCell:
    """Represents an individual cell in an extracted table."""
    row_idx: int
    col_idx: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    bbox: Optional[BoundingBox] = None
    is_header: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_idx": self.row_idx,
            "col_idx": self.col_idx,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "text": self.text,
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "is_header": self.is_header
        }


@dataclass
class TableResult:
    """Represents a structured table extracted from a document page."""
    bbox: Optional[BoundingBox] = None
    num_rows: int = 0
    num_cols: int = 0
    cells: List[TableCell] = field(default_factory=list)
    html: str = ""
    markdown: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox.to_dict() if self.bbox else None,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "cells": [cell.to_dict() for cell in self.cells],
            "html": self.html,
            "markdown": self.markdown
        }


@dataclass
class PageResult:
    """Results for an individual page of a document."""
    page_number: int  # 1-indexed
    width: int
    height: int
    blocks: List[TextBlock] = field(default_factory=list)
    tables: List[TableResult] = field(default_factory=list)
    text: str = ""
    markdown: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "blocks": [b.to_dict() for b in self.blocks],
            "tables": [t.to_dict() for t in self.tables],
            "text": self.text,
            "markdown": self.markdown,
            "latency_ms": round(self.latency_ms, 2)
        }


@dataclass
class OCRResult:
    """Unified result for an entire document processed by AQIVON OmniOCR."""
    document_name: str
    num_pages: int
    pages: List[PageResult] = field(default_factory=list)
    full_text: str = ""
    markdown: str = ""
    engine_used: str = ""
    detected_language: str = "en"
    total_latency_ms: float = 0.0
    vram_usage_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_name": self.document_name,
            "num_pages": self.num_pages,
            "engine_used": self.engine_used,
            "detected_language": self.detected_language,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "vram_usage_mb": round(self.vram_usage_mb, 2) if self.vram_usage_mb else None,
            "full_text": self.full_text,
            "markdown": self.markdown,
            "pages": [p.to_dict() for p in self.pages],
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class EngineCapability:
    """Metadata describing the capabilities and constraints of an OCR engine adapter."""
    engine_id: str
    display_name: str
    supports_handwriting: bool = False
    supports_tables: bool = False
    supports_multipage_pdf: bool = False
    supports_layout_analysis: bool = False
    supported_languages: List[str] = field(default_factory=list)
    gpu_accelerated: bool = True
    typical_vram_requirement_mb: float = 2048.0
    description: str = ""

