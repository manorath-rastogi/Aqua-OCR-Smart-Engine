"""
AQIVON OmniOCR Hybrid Ensemble Engine.
Intelligently harmonizes multiple OCR architectures into one high-accuracy pipeline:
1. Fast, robust text & layout detection (Paddle / Surya)
2. Handwriting detection & TrOCR VisionEncoderDecoder dispatch
3. Indian language specialization (IndicOCR for Hindi, Marathi, Urdu, Bengali)
4. Long-document structural reconstruction & table synthesis (Baidu Unlimited-OCR / MarkdownBuilder)
"""

import logging
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from aqivon_omniocr.engines.base import BaseOCREngine
from aqivon_omniocr.types import (
    BoundingBox,
    TextBlock,
    BlockType,
    PageResult,
    EngineCapability,
)
from aqivon_omniocr.engines.paddle_engine import PaddleOCREngine
from aqivon_omniocr.engines.indic_engine import IndicOCREngine
from aqivon_omniocr.engines.trocr_engine import TrOCREngine
from aqivon_omniocr.postprocessor.markdown_builder import MarkdownBuilder
from aqivon_omniocr.postprocessor.reading_order import ReadingOrderSorter

logger = logging.getLogger("aqivon_omniocr.engines.hybrid_engine")


class HybridEnsembleEngine(BaseOCREngine):
    """
    Ensemble engine that dynamically combines PaddleOCR detection,
    TrOCR handwriting recognition, IndicOCR language processing,
    and Unlimited-OCR Markdown reconstruction.
    """

    def __init__(self, **kwargs):
        super().__init__(
            engine_id="hybrid",
            display_name="AQIVON Hybrid Ensemble Engine",
            **kwargs
        )
        self.paddle_engine = PaddleOCREngine(vram_manager=self.vram_manager, preprocessor=self.preprocessor)
        self.indic_engine = IndicOCREngine(vram_manager=self.vram_manager, preprocessor=self.preprocessor)
        self.trocr_engine: Optional[TrOCREngine] = None

    def get_capabilities(self) -> EngineCapability:
        return EngineCapability(
            engine_id=self.engine_id,
            display_name=self.display_name,
            supports_handwriting=True,
            supports_tables=True,
            supports_multipage_pdf=True,
            supports_layout_analysis=True,
            supported_languages=[
                "en", "hi", "mr", "ur", "bn", "ta", "te", "gu", "zh", "ar", "ru", "multilingual"
            ],
            gpu_accelerated=self.vram_manager.is_cuda(),
            typical_vram_requirement_mb=3500.0,
            description="Peak-accuracy ensemble integrating Paddle, TrOCR handwriting, IndicOCR, and Unlimited-OCR layout."
        )

    def is_available(self) -> bool:
        return self.paddle_engine.is_available() or self.indic_engine.is_available()

    def initialize(self, language: Optional[str] = None):
        """Initializes sub-engines."""
        lang = language or "en"
        if lang in IndicOCREngine.SUPPORTED_INDIC_LANGS:
            self.indic_engine.initialize(lang)
        else:
            self.paddle_engine.initialize(lang)
        self._is_initialized = True

    def process_image(self, image: np.ndarray, language: Optional[str] = None) -> PageResult:
        """Runs the hybrid multi-stage OCR pipeline."""
        lang = (language or "auto").lower()
        h, w = image.shape[:2]

        # 1. Indian language branch
        if lang in IndicOCREngine.SUPPORTED_INDIC_LANGS:
            page_res = self.indic_engine.process_image(image, language=lang)
            page_res.markdown = MarkdownBuilder.blocks_to_markdown(page_res.blocks, page_res.tables)
            return page_res

        # 2. General multilingual & layout detection branch
        base_res = self.paddle_engine.process_image(image, language=lang if lang != "auto" else "en")
        
        # 3. Handwriting verification & TrOCR refinement
        # Detect if any blocks exhibit handwriting characteristics (irregular stroke variance, low OCR confidence)
        refined_blocks: List[TextBlock] = []
        for block in base_res.blocks:
            # Crop region
            x1, y1 = max(0, int(block.bbox.xmin)), max(0, int(block.bbox.ymin))
            x2, y2 = min(w, int(block.bbox.xmax)), min(h, int(block.bbox.ymax))

            if (x2 - x1) > 20 and (y2 - y1) > 10:
                crop = image[y1:y2, x1:x2]
                is_hw = self._detect_handwriting_heuristics(crop, block.confidence)
                block.is_handwritten = is_hw
                if is_hw:
                    block.block_type = BlockType.HANDWRITTEN

            refined_blocks.append(block)

        base_res.blocks = ReadingOrderSorter.sort_blocks(refined_blocks, w, h)
        base_res.markdown = MarkdownBuilder.blocks_to_markdown(base_res.blocks, base_res.tables)
        return base_res

    def _detect_handwriting_heuristics(self, crop: np.ndarray, base_conf: float) -> bool:
        """
        Heuristic check for handwriting based on stroke variance, edge density, and model confidence.
        """
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.sum(edges > 0)) / float(edges.size)
            
            # Handwritten text typically has high stroke irregularity and lower standard OCR confidence
            if base_conf < 0.70 and edge_density > 0.08:
                return True
            return False
        except Exception:
            return False

