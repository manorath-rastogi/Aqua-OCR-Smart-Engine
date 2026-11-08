"""
AQIVON OmniOCR Surya Layout & OCR Adapter.
Specialized in complex document layout analysis, column segmentation,
reading order estimation, and multilingual line detection across 90+ languages.
"""

import logging
from typing import Any, List, Optional
import numpy as np
from PIL import Image

from aqivon_omniocr.engines.base import BaseOCREngine
from aqivon_omniocr.types import (
    BoundingBox,
    TextBlock,
    BlockType,
    PageResult,
    EngineCapability,
)

logger = logging.getLogger("aqivon_omniocr.engines.surya_engine")


class SuryaOCREngine(BaseOCREngine):
    """
    Adapter for Surya OCR and document layout models.
    Excels at multi-column academic papers, text line segmentation, and reading order.
    """

    def __init__(self, **kwargs):
        super().__init__(
            engine_id="surya",
            display_name="Surya Document Layout & Line Engine",
            **kwargs
        )
        self.det_model: Optional[Any] = None
        self.det_processor: Optional[Any] = None
        self.rec_model: Optional[Any] = None
        self.rec_processor: Optional[Any] = None

    def get_capabilities(self) -> EngineCapability:
        return EngineCapability(
            engine_id=self.engine_id,
            display_name=self.display_name,
            supports_handwriting=False,
            supports_tables=True,
            supports_multipage_pdf=True,
            supports_layout_analysis=True,
            supported_languages=["en", "hi", "mr", "bn", "ur", "multilingual"],
            gpu_accelerated=self.vram_manager.is_cuda(),
            typical_vram_requirement_mb=2500.0,
            description="Transformer-based layout detection, reading order, and multilingual line-level OCR."
        )

    def is_available(self) -> bool:
        try:
            import surya
            return True
        except ImportError:
            return False

    def initialize(self):
        """Initializes Surya detection and recognition models."""
        if not self.is_available():
            logger.info("Surya package not installed in environment; using layout fallback.")
            self._is_initialized = True
            return

        try:
            from surya.detection import load_model as load_det_model, load_processor as load_det_proc
            from surya.recognition import load_model as load_rec_model, load_processor as load_rec_proc

            logger.info("Loading Surya models...")
            self.det_processor = load_det_proc()
            self.det_model = load_det_model()
            self.rec_processor = load_rec_proc()
            self.rec_model = load_rec_model()

            self._is_initialized = True
        except Exception as e:
            logger.warning("Failed to initialize Surya models (%s).", e)
            self._is_initialized = True

    def process_image(self, image: np.ndarray, language: Optional[str] = None) -> PageResult:
        """Processes an image using Surya or layout-aware fallback."""
        if not self._is_initialized:
            self.initialize()

        h, w = image.shape[:2]

        if self.det_model is not None and self.rec_model is not None:
            try:
                from surya.ocr import run_ocr
                pil_img = Image.fromarray(image)
                predictions = run_ocr(
                    [pil_img],
                    [language or "en"],
                    self.det_model,
                    self.det_processor,
                    self.rec_model,
                    self.rec_processor
                )
                blocks: List[TextBlock] = []
                if predictions and len(predictions) > 0:
                    pred = predictions[0]
                    for line in pred.text_lines:
                        # line.bbox is [xmin, ymin, xmax, ymax]
                        bbox = BoundingBox(
                            xmin=float(line.bbox[0]),
                            ymin=float(line.bbox[1]),
                            xmax=float(line.bbox[2]),
                            ymax=float(line.bbox[3])
                        )
                        blocks.append(TextBlock(
                            text=line.text.strip(),
                            bbox=bbox,
                            confidence=float(getattr(line, "confidence", 0.95)),
                            language=language,
                            block_type=BlockType.PARAGRAPH
                        ))
                return PageResult(page_number=1, width=w, height=h, blocks=blocks)
            except Exception as e:
                logger.error("Surya execution failed (%s); delegating to fallback.", e)

        # Fallback to high-performance multilingual layout engine
        from aqivon_omniocr.engines.paddle_engine import PaddleOCREngine
        fallback = PaddleOCREngine(
            vram_manager=self.vram_manager,
            preprocessor=self.preprocessor
        )
        return fallback.process_image(image, language=language)

