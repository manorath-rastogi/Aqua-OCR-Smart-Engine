"""
Aqua OCR Smart Engine Engines Package.
Contains modular engine adapters for all integrated OCR models and pipelines.
"""

from aqua_ocr_smart_engine.engines.base import BaseOCREngine
from aqua_ocr_smart_engine.engines.paddle_engine import PaddleOCREngine
from aqua_ocr_smart_engine.engines.trocr_engine import TrOCREngine
from aqua_ocr_smart_engine.engines.indic_engine import IndicOCREngine
from aqua_ocr_smart_engine.engines.unlimited_ocr import UnlimitedOCREngine
from aqua_ocr_smart_engine.engines.surya_engine import SuryaOCREngine
from aqua_ocr_smart_engine.engines.doctr_engine import DocTREngine
from aqua_ocr_smart_engine.engines.mineru_engine import MinerUEngine
from aqua_ocr_smart_engine.engines.tesseract_engine import TesseractEngine
from aqua_ocr_smart_engine.engines.hybrid_engine import HybridEnsembleEngine

__all__ = [
    "BaseOCREngine",
    "PaddleOCREngine",
    "TrOCREngine",
    "IndicOCREngine",
    "UnlimitedOCREngine",
    "SuryaOCREngine",
    "DocTREngine",
    "MinerUEngine",
    "TesseractEngine",
    "HybridEnsembleEngine",
]
