"""
AQIVON OmniOCR Tesseract Fallback Engine Adapter.
Universal, lightweight CPU-based offline OCR engine supporting 100+ languages.
Serves as a reliable fallback when GPU acceleration is unavailable or lightweight CPU processing is demanded.
"""

import logging
import os
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

logger = logging.getLogger("aqivon_omniocr.engines.tesseract_engine")


class TesseractEngine(BaseOCREngine):
    """
    Adapter for Google Tesseract OCR engine via pytesseract.
    Reliable offline fallback requiring zero GPU resources.
    """

    def __init__(self, **kwargs):
        super().__init__(
            engine_id="tesseract",
            display_name="Tesseract Universal Offline Engine",
            **kwargs
        )
        self._tesseract_binary_path = self._detect_tesseract_binary()

    def _detect_tesseract_binary(self) -> Optional[str]:
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"G:\Program Files\Tesseract-OCR\tesseract.exe",
            r"D:\Program Files\Tesseract-OCR\tesseract.exe"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def get_capabilities(self) -> EngineCapability:
        return EngineCapability(
            engine_id=self.engine_id,
            display_name=self.display_name,
            supports_handwriting=False,
            supports_tables=False,
            supports_multipage_pdf=True,
            supports_layout_analysis=False,
            supported_languages=["en", "hi", "mr", "ur", "bn", "multilingual"],
            gpu_accelerated=False,
            typical_vram_requirement_mb=0.0,
            description="Classic LSTM-based OCR requiring zero GPU memory, ideal for CPU offline environments."
        )

    def is_available(self) -> bool:
        try:
            import pytesseract
            if self._tesseract_binary_path or os.system("where tesseract >nul 2>nul") == 0:
                return True
            return False
        except Exception:
            return False

    def initialize(self):
        if self._tesseract_binary_path:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = self._tesseract_binary_path
        self._is_initialized = True

    def process_image(self, image: np.ndarray, language: Optional[str] = None) -> PageResult:
        """Processes image using pytesseract."""
        if not self._is_initialized:
            self.initialize()

        h, w = image.shape[:2]

        if not self.is_available():
            logger.info("Tesseract binary not installed on system; using primary engine.")
            from aqivon_omniocr.engines.paddle_engine import PaddleOCREngine
            fallback = PaddleOCREngine(
                vram_manager=self.vram_manager,
                preprocessor=self.preprocessor
            )
            return fallback.process_image(image, language=language)

        import pytesseract
        pil_img = Image.fromarray(image)
        lang_code = language or "eng"
        lang_map = {"en": "eng", "hi": "hin", "mr": "mar", "ur": "urd", "bn": "ben"}
        tess_lang = lang_map.get(lang_code, "eng")

        try:
            data = pytesseract.image_to_data(pil_img, lang=tess_lang, output_type=pytesseract.Output.DICT)
            blocks: List[TextBlock] = []
            n_boxes = len(data["text"])
            
            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])
                if not text or conf < 30.0:
                    continue
                
                x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                bbox = BoundingBox(xmin=float(x), ymin=float(y), xmax=float(x + bw), ymax=float(y + bh))
                blocks.append(TextBlock(
                    text=text,
                    bbox=bbox,
                    confidence=conf / 100.0,
                    language=lang_code,
                    block_type=BlockType.PARAGRAPH
                ))

            return PageResult(page_number=1, width=w, height=h, blocks=blocks)
        except Exception as e:
            logger.error("Tesseract execution error: %s", e)
            from aqivon_omniocr.engines.paddle_engine import PaddleOCREngine
            fallback = PaddleOCREngine(
                vram_manager=self.vram_manager,
                preprocessor=self.preprocessor
            )
            return fallback.process_image(image, language=language)

