"""
AQIVON OmniOCR docTR Adapter.
Seamless integration of Mindee docTR end-to-end deep learning OCR (PyTorch backend).
Optimized for invoices, receipts, corporate documents, and form parsing.
"""

import logging
from typing import Any, List, Optional
import numpy as np

from aqivon_omniocr.engines.base import BaseOCREngine
from aqivon_omniocr.types import (
    BoundingBox,
    TextBlock,
    BlockType,
    PageResult,
    EngineCapability,
)

logger = logging.getLogger("aqivon_omniocr.engines.doctr_engine")


class DocTREngine(BaseOCREngine):
    """
    Adapter for Mindee docTR OCR models with PyTorch backbones.
    High accuracy for documents, receipts, invoices, and structured forms.
    """

    def __init__(self, **kwargs):
        super().__init__(
            engine_id="doctr",
            display_name="docTR End-to-End Deep Learning Engine",
            **kwargs
        )
        self.predictor: Optional[Any] = None

    def get_capabilities(self) -> EngineCapability:
        return EngineCapability(
            engine_id=self.engine_id,
            display_name=self.display_name,
            supports_handwriting=False,
            supports_tables=False,
            supports_multipage_pdf=True,
            supports_layout_analysis=True,
            supported_languages=["en", "fr", "de", "es", "it", "pt"],
            gpu_accelerated=self.vram_manager.is_cuda(),
            typical_vram_requirement_mb=2000.0,
            description="PyTorch end-to-end document OCR with DBNet text detection and CRNN/SAR recognition."
        )

    def is_available(self) -> bool:
        try:
            import doctr
            return True
        except ImportError:
            return False

    def initialize(self):
        """Initializes docTR PyTorch predictor."""
        if not self.is_available():
            logger.info("docTR is not installed in the environment; using fallback.")
            self._is_initialized = True
            return

        try:
            from doctr.models import ocr_predictor
            device = self.vram_manager.get_device()
            logger.info("Initializing docTR predictor on %s...", device)
            self.predictor = ocr_predictor(pretrained=True)
            if self.vram_manager.is_cuda():
                self.predictor.cuda()
            self._is_initialized = True
        except Exception as e:
            logger.warning("Failed to initialize docTR predictor: %s", e)
            self._is_initialized = True

    def process_image(self, image: np.ndarray, language: Optional[str] = None) -> PageResult:
        """Processes an image using docTR or fallback."""
        if not self._is_initialized:
            self.initialize()

        h, w = image.shape[:2]

        if self.predictor is not None:
            try:
                # docTR expects list of numpy arrays (RGB)
                doc = [image]
                result = self.predictor(doc)
                blocks: List[TextBlock] = []

                for page in result.pages:
                    for block in page.blocks:
                        for line in block.lines:
                            line_text = " ".join([word.value for word in line.words]).strip()
                            if not line_text:
                                continue
                            # docTR coords are relative (0.0 to 1.0) [(ymin, xmin), (ymax, xmax)]
                            (ymin, xmin), (ymax, xmax) = line.geometry
                            bbox = BoundingBox(
                                xmin=float(xmin * w),
                                ymin=float(ymin * h),
                                xmax=float(xmax * w),
                                ymax=float(ymax * h)
                            )
                            avg_conf = sum(w.confidence for w in line.words) / len(line.words) if line.words else 0.9
                            blocks.append(TextBlock(
                                text=line_text,
                                bbox=bbox,
                                confidence=float(avg_conf),
                                language=language or "en",
                                block_type=BlockType.PARAGRAPH
                            ))
                return PageResult(page_number=1, width=w, height=h, blocks=blocks)
            except Exception as e:
                logger.error("docTR execution error: %s", e)

        # Fallback to high-performance multilingual engine
        from aqivon_omniocr.engines.paddle_engine import PaddleOCREngine
        fallback = PaddleOCREngine(
            vram_manager=self.vram_manager,
            preprocessor=self.preprocessor
        )
        return fallback.process_image(image, language=language)

