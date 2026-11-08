"""
AQIVON OmniOCR Engines Package.
Contains modular engine adapters for all integrated OCR models and pipelines.
"""

from aqivon_omniocr.engines.base import BaseOCREngine
from aqivon_omniocr.engines.paddle_engine import PaddleOCREngine
from aqivon_omniocr.engines.trocr_engine import TrOCREngine
from aqivon_omniocr.engines.indic_engine import IndicOCREngine
from aqivon_omniocr.engines.unlimited_ocr import UnlimitedOCREngine
from aqivon_omniocr.engines.surya_engine import SuryaOCREngine
from aqivon_omniocr.engines.doctr_engine import DocTREngine
from aqivon_omniocr.engines.mineru_engine import MinerUEngine
from aqivon_omniocr.engines.tesseract_engine import TesseractEngine
from aqivon_omniocr.engines.hybrid_engine import HybridEnsembleEngine

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

