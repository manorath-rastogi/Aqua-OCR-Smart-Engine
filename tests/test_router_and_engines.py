"""
Unit & Integration tests for Aqua OCR Smart Engine Intelligent Router, Engines, and Metrics.
"""

import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aqua_ocr_smart_engine.router import IntelligentRouter
from aqua_ocr_smart_engine.types import BoundingBox, TextBlock, BlockType, PageResult
from aqua_ocr_smart_engine.utils.metrics import OCRMetrics
from aqua_ocr_smart_engine.utils.visualizer import LayoutVisualizer


def test_metrics_calculation():
    ref = "Aqua OCR Smart Engine production ready platform"
    hyp = "Aqua OCR Smart Engine production ready platform"
    assert OCRMetrics.compute_cer(ref, hyp) == 0.0
    assert OCRMetrics.compute_wer(ref, hyp) == 0.0

    # 1 char error
    hyp_err = "Aqua OCR Smart Engine production ready platfomm"
    cer = OCRMetrics.compute_cer(ref, hyp_err)
    assert 0.0 < cer < 0.1

    summary = OCRMetrics.compute_benchmark_summary([{"reference": ref, "prediction": hyp}])
    assert summary["char_accuracy_pct"] == 100.0


def test_router_engine_selection():
    router = IntelligentRouter()

    # Capabilities
    caps = router.list_engines()
    assert len(caps) >= 8

    # Routing logic tests
    eng_hw = router._select_engine(requested_engine=None, language="en", is_handwritten=True)
    assert eng_hw == "trocr_handwritten"

    eng_hi = router._select_engine(requested_engine=None, language="hi", is_handwritten=False)
    assert eng_hi == "indicocr"

    eng_mr = router._select_engine(requested_engine=None, language="mr", is_handwritten=False)
    assert eng_mr == "indicocr"

    eng_ur = router._select_engine(requested_engine=None, language="ur", is_handwritten=False)
    assert eng_ur == "indicocr"

    eng_explicit = router._select_engine(requested_engine="unlimited_ocr", language="en", is_handwritten=False)
    assert eng_explicit == "unlimited_ocr"


def test_visualizer():
    # Blank 300x400 image
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    block = TextBlock(
        text="Sample Line",
        bbox=BoundingBox(50, 50, 200, 90),
        confidence=0.98,
        block_type=BlockType.HEADING_1,
        reading_order_idx=0
    )
    page_res = PageResult(1, 400, 300, blocks=[block])
    annotated = LayoutVisualizer.draw_page_result(img, page_res)
    assert annotated.shape == (300, 400, 3)
    # Check that pixels were drawn (not all 255)
    assert np.any(annotated != 255)


if __name__ == "__main__":
    test_metrics_calculation()
    test_router_engine_selection()
    test_visualizer()
    print("Router, Metrics, and Visualizer tests passed successfully!")

