"""
Unit tests for AQIVON OmniOCR foundation components:
types, config, vram_manager, and preprocessor.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from aqivon_omniocr.config import OmniOCRConfig, DEFAULT_CONFIG
from aqivon_omniocr.types import (
    BoundingBox,
    TextBlock,
    TableCell,
    TableResult,
    PageResult,
    OCRResult,
    BlockType
)
from aqivon_omniocr.vram_manager import VRAMManager
from aqivon_omniocr.preprocessor import DocumentPreprocessor


def test_bounding_box():
    bbox = BoundingBox(xmin=10.0, ymin=20.0, xmax=110.0, ymax=120.0)
    assert bbox.width == 100.0
    assert bbox.height == 100.0
    assert bbox.center == (60.0, 70.0)
    assert bbox.area == 10000.0
    assert bbox.to_rect_list() == [10.0, 20.0, 110.0, 120.0]

    # Polygon constructor
    poly = [(10, 20), (110, 20), (110, 120), (10, 120)]
    bbox_poly = BoundingBox.from_polygon(poly)
    assert bbox_poly.xmin == 10.0
    assert bbox_poly.ymax == 120.0


def test_ocr_result_serialization():
    bbox = BoundingBox(10, 20, 100, 50)
    block = TextBlock(
        text="AQIVON OmniOCR Header",
        bbox=bbox,
        confidence=0.99,
        block_type=BlockType.HEADING_1
    )
    page = PageResult(
        page_number=1,
        width=800,
        height=1200,
        blocks=[block],
        text="AQIVON OmniOCR Header",
        markdown="# AQIVON OmniOCR Header\n"
    )
    doc = OCRResult(
        document_name="test_doc.png",
        num_pages=1,
        pages=[page],
        full_text="AQIVON OmniOCR Header",
        markdown="# AQIVON OmniOCR Header\n",
        engine_used="test_engine"
    )
    doc_json = doc.to_json()
    assert "AQIVON OmniOCR Header" in doc_json
    assert "test_engine" in doc_json


def test_vram_manager():
    vm = VRAMManager(max_vram_gb=11.0, default_precision="fp16")
    device = vm.get_device()
    assert device in ["cpu"] or "cuda" in device
    stats = vm.get_vram_stats()
    assert "allocated_mb" in stats
    vm.clear_cache()


def test_document_preprocessor_synthetic():
    preprocessor = DocumentPreprocessor()
    
    # Generate synthetic image (white background with black text-like bar)
    img_arr = np.ones((400, 600, 3), dtype=np.uint8) * 255
    img_arr[180:220, 100:500] = 0  # Black bar
    
    pages = preprocessor.load_document(img_arr)
    assert len(pages) == 1
    assert pages[0].shape == (400, 600, 3)

    # PIL Image input
    pil_img = Image.fromarray(img_arr)
    pages_pil = preprocessor.load_document(pil_img)
    assert len(pages_pil) == 1


if __name__ == "__main__":
    test_bounding_box()
    test_ocr_result_serialization()
    test_vram_manager()
    test_document_preprocessor_synthetic()
    print("All foundation tests passed successfully!")
