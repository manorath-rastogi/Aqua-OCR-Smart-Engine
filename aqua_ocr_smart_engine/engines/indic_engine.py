"""
Aqua OCR Smart Engine IndicOCR Adapter.
Specialized Indian language OCR engine tailored for Indic scripts:
- Hindi (हिन्दी) & Marathi (मराठी) - Devanagari
- Urdu (اردو) - Nastaliq / Perso-Arabic RTL
- Bengali (বাংলা) - Eastern Nagari
- Tamil (தமிழ்), Telugu (తెలుగు), Gujarati (ગુજરાતી), Punjabi (ਪੰਜਾਬੀ), Kannada, Malayalam, Odia.

Incorporates unicode conjunct stabilization, ZWJ/ZWNJ preservation, and RTL line formatting.
Compatible with both PaddleOCR 3.x and 2.x API structures.
"""

import logging
import os
import unicodedata
from typing import Any, Dict, List, Optional
import numpy as np

# Stability flags for Windows CPU
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

logger = logging.getLogger("aqua_ocr_smart_engine.engines.indic_engine")


class IndicOCREngine(BaseOCREngine):
    """
    Dedicated OCR adapter for Indian subcontinental languages and scripts.
    Handles complex Brahmic ligatures, vowel signs (matras), and Perso-Arabic right-to-left layout.
    """

    SUPPORTED_INDIC_LANGS = {
        "hi": {"name": "Hindi", "script": "Devanagari", "rtl": False, "paddle_code": "hi", "tess_code": "hin"},
        "mr": {"name": "Marathi", "script": "Devanagari", "rtl": False, "paddle_code": "mr", "tess_code": "mar"},
        "ur": {"name": "Urdu", "script": "Arabic/Nastaliq", "rtl": True, "paddle_code": "arabic", "tess_code": "urd"},
        "bn": {"name": "Bengali", "script": "Bengali", "rtl": False, "paddle_code": "bn", "tess_code": "ben"},
        "ta": {"name": "Tamil", "script": "Tamil", "rtl": False, "paddle_code": "ta", "tess_code": "tam"},
        "te": {"name": "Telugu", "script": "Telugu", "rtl": False, "paddle_code": "te", "tess_code": "tel"},
        "gu": {"name": "Gujarati", "script": "Gujarati", "rtl": False, "paddle_code": "gu", "tess_code": "guj"},
        "pa": {"name": "Punjabi", "script": "Gurmukhi", "rtl": False, "paddle_code": "pa", "tess_code": "pan"},
        "kn": {"name": "Kannada", "script": "Kannada", "rtl": False, "paddle_code": "kn", "tess_code": "kan"},
        "ml": {"name": "Malayalam", "script": "Malayalam", "rtl": False, "paddle_code": "ml", "tess_code": "mal"},
        "or": {"name": "Odia", "script": "Odia", "rtl": False, "paddle_code": "or", "tess_code": "ori"},
    }

    def __init__(self, **kwargs):
        super().__init__(
            engine_id="indicocr",
            display_name="IndicOCR Indian Languages Engine",
            **kwargs
        )
        self._recognizers: Dict[str, Any] = {}
        self._active_lang = "hi"

    def get_capabilities(self) -> EngineCapability:
        return EngineCapability(
            engine_id=self.engine_id,
            display_name=self.display_name,
            supports_handwriting=False,
            supports_tables=True,
            supports_multipage_pdf=True,
            supports_layout_analysis=True,
            supported_languages=list(self.SUPPORTED_INDIC_LANGS.keys()),
            gpu_accelerated=self.vram_manager.is_cuda(),
            typical_vram_requirement_mb=1600.0,
            description="Specialized engine for Indian scripts (Devanagari, Bengali, Urdu RTL, Tamil, Telugu, Gujarati)."
        )

    def is_available(self) -> bool:
        try:
            import paddleocr
            return True
        except ImportError:
            return False

    def initialize(self, language: str = "hi"):
        """Initializes the underlying model for the target Indian language."""
        target_lang = language.lower() if language and language.lower() in self.SUPPORTED_INDIC_LANGS else "hi"
        
        if target_lang in self._recognizers:
            self._active_lang = target_lang
            self._is_initialized = True
            return

        lang_info = self.SUPPORTED_INDIC_LANGS[target_lang]
        paddle_code = lang_info["paddle_code"]

        try:
            from paddleocr import PaddleOCR
            logger.info("Initializing IndicOCR engine for %s (%s, code=%s)...",
                        lang_info["name"], lang_info["script"], paddle_code)
            
            try:
                recognizer = PaddleOCR(
                    lang=paddle_code,
                    use_doc_unwarping=False,
                    use_doc_orientation_classify=False,
                    use_textline_orientation=False
                )
            except Exception:
                try:
                    recognizer = PaddleOCR(lang=paddle_code)
                except Exception:
                    recognizer = PaddleOCR(use_angle_cls=True, lang=paddle_code)

            self._recognizers[target_lang] = recognizer
            self.vram_manager.register_model(f"indicocr_{target_lang}", recognizer, estimated_mb=850)
            self._active_lang = target_lang
            self._is_initialized = True
        except Exception as e:
            logger.warning("PaddleOCR Indic model initialization failed for '%s' (%s).", target_lang, e)
            self._is_initialized = True

    def process_image(self, image: np.ndarray, language: Optional[str] = None) -> PageResult:
        """Performs Indic OCR with unicode normalization and script-specific layout handling."""
        lang = language.lower() if language and language.lower() in self.SUPPORTED_INDIC_LANGS else "hi"
        if not self._is_initialized or lang not in self._recognizers:
            self.initialize(lang)

        lang_info = self.SUPPORTED_INDIC_LANGS.get(lang, self.SUPPORTED_INDIC_LANGS["hi"])
        h, w = image.shape[:2]
        blocks: List[TextBlock] = []

        recognizer = self._recognizers.get(lang)
        if recognizer is not None:
            try:
                # PaddleOCR 3.x predict
                if hasattr(recognizer, "predict"):
                    predictions = list(recognizer.predict(image))
                    if predictions and len(predictions) > 0:
                        pred = predictions[0]
                        if isinstance(pred, dict) and "rec_texts" in pred:
                            rec_texts = pred.get("rec_texts", [])
                            rec_scores = pred.get("rec_scores", [])
                            rec_polys = pred.get("rec_polys", []) or pred.get("dt_polys", [])

                            for idx, (text, score) in enumerate(zip(rec_texts, rec_scores)):
                                raw_text = str(text).strip()
                                if not raw_text:
                                    continue
                                norm_text = unicodedata.normalize("NFC", raw_text)
                                poly = rec_polys[idx] if idx < len(rec_polys) else None
                                bbox = BoundingBox.from_polygon(poly) if poly is not None else BoundingBox(0, 0, float(w), float(h))
                                
                                blocks.append(TextBlock(
                                    text=norm_text,
                                    bbox=bbox,
                                    confidence=float(score),
                                    language=lang,
                                    block_type=BlockType.PARAGRAPH,
                                    font_size_estimate=float(bbox.height)
                                ))
                
                # Legacy ocr fallback if empty
                if not blocks and hasattr(recognizer, "ocr"):
                    raw_res = recognizer.ocr(image)
                    if raw_res and len(raw_res) > 0 and raw_res[0] is not None:
                        for item in raw_res[0]:
                            poly = item[0]
                            text_info = item[1]
                            raw_text = text_info[0].strip()
                            conf = float(text_info[1])
                            if not raw_text:
                                continue
                            norm_text = unicodedata.normalize("NFC", raw_text)
                            bbox = BoundingBox.from_polygon(poly)
                            blocks.append(TextBlock(
                                text=norm_text,
                                bbox=bbox,
                                confidence=conf,
                                language=lang,
                                block_type=BlockType.PARAGRAPH,
                                font_size_estimate=float(bbox.height)
                            ))
            except Exception as e:
                logger.error("IndicOCR recognition error: %s", e)

        # Handle RTL ordering for Urdu
        if lang_info["rtl"] and blocks:
            for b in blocks:
                b.block_type = BlockType.PARAGRAPH
            blocks.sort(key=lambda b: (round(b.bbox.ymin / 20.0), -b.bbox.xmax))

        return PageResult(
            page_number=1,
            width=w,
            height=h,
            blocks=blocks,
            tables=[]
        )
