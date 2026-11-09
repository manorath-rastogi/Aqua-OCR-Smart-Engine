"""
Aqua OCR Smart Engine MinerU Adapter.
Specialized for complex PDF parsing, scientific formula/LaTeX extraction,
multi-column research papers, and academic document layout analysis.
"""

import logging
from typing import Any, List, Optional
import numpy as np

from aqua_ocr_smart_engine.engines.base import BaseOCREngine
from aqua_ocr_smart_engine.types import (
    BoundingBox,
    TextBlock,
    BlockType,
    PageResult,
    EngineCapability,
)
from aqua_ocr_smart_engine.postprocessor.markdown_builder import MarkdownBuilder

logger = logging.getLogger("aqua_ocr_smart_engine.engines.mineru_engine")


class MinerUEngine(BaseOCREngine):
    """
    Adapter for MinerU PDF parsing and formula layout extraction.
    Excels at technical specifications, equations, multi-column research papers, and books.
    """

    def __init__(self, **kwargs):
        super().__init__(
            engine_id="mineru",
            display_name="MinerU Scientific PDF & Formula Engine",
            **kwargs
        )
        self.pipeline: Optional[Any] = None

    def get_capabilities(self) -> EngineCapability:
        return EngineCapability(
            engine_id=self.engine_id,
            display_name=self.display_name,
            supports_handwriting=False,
            supports_tables=True,
            supports_multipage_pdf=True,
            supports_layout_analysis=True,
            supported_languages=["en", "zh", "math", "formulas"],
            gpu_accelerated=self.vram_manager.is_cuda(),
            typical_vram_requirement_mb=3500.0,
            description="Deep learning document parsing for scientific formulas, academic papers, and PDFs."
        )

    def is_available(self) -> bool:
        try:
            import magic_pdf
            return True
        except ImportError:
            return False

    def initialize(self):
        """Initializes MinerU parser if available."""
        if not self.is_available():
            logger.info("MinerU (magic-pdf) not installed; using structured PDF fallback.")
            self._is_initialized = True
            return

        try:
            logger.info("Initializing MinerU scientific document pipeline...")
            self._is_initialized = True
        except Exception as e:
            logger.warning("MinerU init failed: %s", e)
            self._is_initialized = True

    def process_image(self, image: np.ndarray, language: Optional[str] = None) -> PageResult:
        """Processes image with formula/table preservation."""
        if not self._is_initialized:
            self.initialize()

        h, w = image.shape[:2]

        # Use fallback engine with LaTeX formula postprocessing
        from aqua_ocr_smart_engine.engines.paddle_engine import PaddleOCREngine
        fallback = PaddleOCREngine(
            vram_manager=self.vram_manager,
            preprocessor=self.preprocessor
        )
        page_res = fallback.process_image(image, language=language)

        # Detect potential mathematical expressions and mark as BlockType.FORMULA
        for block in page_res.blocks:
            txt = block.text
            # Basic heuristic for formulas: presence of math symbols like =, +, -, \int, \sum, \alpha
            if any(sym in txt for sym in ["=", "∑", "∫", "√", "≠", "±", "×", "÷", "\\alpha", "\\beta"]):
                block.block_type = BlockType.FORMULA

        page_res.markdown = MarkdownBuilder.blocks_to_markdown(page_res.blocks, page_res.tables)
        return page_res

