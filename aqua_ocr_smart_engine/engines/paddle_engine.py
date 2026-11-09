"""
Aqua OCR Smart Engine PaddleOCR Adapter.
High-throughput, multilingual OCR engine leveraging DBNet detection and SVTR/CRNN recognition.
Supports 80+ languages including English, Hindi, Marathi, Arabic, Chinese, and European scripts.
Compatible with both modern PaddleOCR 3.x and 2.x API formats with Windows CPU/GPU stability.
"""

import logging
import os
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure Windows CPU oneDNN/PIR stability for Paddle 3.x
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from aqua_ocr_smart_engine.engines.base import BaseOCREngine
from aqua_ocr_smart_engine.types import (
    BoundingBox,
    TextBlock,
    BlockType,
    PageResult,
    EngineCapability,
)

logger = logging.getLogger("aqua_ocr_smart_engine.engines.paddle_engine")


class PaddleOCREngine(BaseOCREngine):
    """Adapter for PaddleOCR detection, direction classifier, and recognition pipelines."""

    def __init__(self, **kwargs):
        super().__init__(
            engine_id="paddleocr",
            display_name="PaddleOCR Multilingual Engine",
            **kwargs
        )
        self._ocr_instances: Dict[str, Any] = {}
        self._current_lang = "en"

    def get_capabilities(self) -> EngineCapability:
        return EngineCapability(
            engine_id=self.engine_id,
            display_name=self.display_name,
            supports_handwriting=False,
            supports_tables=True,
            supports_multipage_pdf=True,
            supports_layout_analysis=True,
            supported_languages=[
                "en", "hi", "mr", "zh", "ar", "ru", "es", "fr", "de", "ja", "ko"
            ],
            gpu_accelerated=self.vram_manager.is_cuda(),
            typical_vram_requirement_mb=1500.0,
            description="Ultra-fast multilingual text detection and recognition with angle classification."
        )

    def is_available(self) -> bool:
        try:
            import paddleocr
            return True
        except ImportError:
            return False

    def initialize(self, language: str = "en"):
        """Initializes PaddleOCR instance for specified language."""
        if not self.is_available():
            raise RuntimeError("PaddleOCR is not installed in the environment.")

        # Map language codes to PaddleOCR internal codes
        lang_map = {
            "en": "en",
            "hi": "hi",
            "mr": "mr",
            "zh": "ch",
            "ar": "arabic",
            "ru": "ru",
            "es": "es",
            "fr": "fr",
            "de": "german",
            "ja": "japan",
            "ko": "korean",
            "auto": "en"
        }
        target_lang = lang_map.get(language, "en")
        
        if target_lang not in self._ocr_instances:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR (lang=%s)...", target_lang)
            
            try:
                # PaddleOCR 3.x initialization with robust document flags
                ocr_instance = PaddleOCR(
                    lang=target_lang,
                    use_doc_unwarping=False,
                    use_doc_orientation_classify=False,
                    use_textline_orientation=False
                )
            except Exception as e:
                logger.debug("PaddleOCR 3.x init failed (%s), trying legacy init...", e)
                try:
                    ocr_instance = PaddleOCR(lang=target_lang)
                except Exception:
                    ocr_instance = PaddleOCR(use_angle_cls=True, lang=target_lang)

            self._ocr_instances[target_lang] = ocr_instance
            self.vram_manager.register_model(f"paddleocr_{target_lang}", ocr_instance, estimated_mb=800)

        self._current_lang = target_lang
        self._is_initialized = True

    def process_image(self, image: np.ndarray, language: Optional[str] = None) -> PageResult:
        """Runs PaddleOCR on an RGB image."""
        lang = language or "en"
        if not self._is_initialized or lang not in self._ocr_instances:
            self.initialize(lang)

        ocr_instance = self._ocr_instances.get(self._current_lang) or self._ocr_instances.get("en")
        h, w = image.shape[:2]

        blocks: List[TextBlock] = []

        try:
            # Try PaddleOCR 3.x predict() pipeline first
            if hasattr(ocr_instance, "predict"):
                predictions = list(ocr_instance.predict(image))
                if predictions and len(predictions) > 0:
                    pred = predictions[0]
                    # PaddleOCR 3.x dict format
                    if isinstance(pred, dict) and "rec_texts" in pred:
                        rec_texts = pred.get("rec_texts", [])
                        rec_scores = pred.get("rec_scores", [])
                        rec_polys = pred.get("rec_polys", []) or pred.get("dt_polys", [])

                        # Calculate median height for heading classification
                        heights = []
                        for poly in rec_polys:
                            ys = [p[1] for p in poly]
                            heights.append(float(max(ys) - min(ys)))
                        median_h = np.median(heights) if heights else 20.0

                        for idx, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                            text_clean = str(text).strip()
                            if not text_clean:
                                continue

                            poly = rec_polys[idx] if idx < len(rec_polys) else None
                            if poly is not None:
                                bbox = BoundingBox.from_polygon(poly)
                            else:
                                bbox = BoundingBox(0, 0, float(w), float(h))

                            # Heading classification
                            b_height = bbox.height
                            block_type = BlockType.PARAGRAPH
                            if b_height > (median_h * 1.8):
                                block_type = BlockType.HEADING_1
                            elif b_height > (median_h * 1.4):
                                block_type = BlockType.HEADING_2
                            elif b_height > (median_h * 1.2):
                                block_type = BlockType.HEADING_3

                            blocks.append(TextBlock(
                                text=text_clean,
                                bbox=bbox,
                                confidence=float(score),
                                language=lang,
                                block_type=block_type,
                                font_size_estimate=float(b_height)
                            ))
                        return PageResult(page_number=1, width=w, height=h, blocks=blocks)

            # Fallback to legacy ocr() method
            raw_res = ocr_instance.ocr(image)
            if raw_res and len(raw_res) > 0 and raw_res[0] is not None:
                heights = []
                for item in raw_res[0]:
                    poly = item[0]
                    ys = [p[1] for p in poly]
                    heights.append(max(ys) - min(ys))
                median_height = np.median(heights) if heights else 20.0

                for item in raw_res[0]:
                    poly = item[0]
                    text_info = item[1]
                    text = text_info[0].strip()
                    conf = float(text_info[1])
                    if not text:
                        continue
                    bbox = BoundingBox.from_polygon(poly)
                    block_height = bbox.height

                    block_type = BlockType.PARAGRAPH
                    if block_height > (median_height * 1.8):
                        block_type = BlockType.HEADING_1
                    elif block_height > (median_height * 1.4):
                        block_type = BlockType.HEADING_2
                    elif block_height > (median_height * 1.2):
                        block_type = BlockType.HEADING_3

                    blocks.append(TextBlock(
                        text=text,
                        bbox=bbox,
                        confidence=conf,
                        language=lang,
                        block_type=block_type,
                        font_size_estimate=float(block_height)
                    ))

        except Exception as e:
            logger.error("PaddleOCR execution error: %s", e)

        return PageResult(
            page_number=1,
            width=w,
            height=h,
            blocks=blocks,
            tables=[]
        )
