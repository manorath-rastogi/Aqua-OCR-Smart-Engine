"""
Aqua OCR Smart Engine
A unified, production-ready AI OCR platform by AQIVON Technologies.
"""

from aqua_ocr_smart_engine.config import AquaOCRConfig, OmniOCRConfig, DEFAULT_CONFIG
from aqua_ocr_smart_engine.types import (
    BoundingBox,
    TextBlock,
    TableCell,
    TableResult,
    PageResult,
    OCRResult,
    BlockType,
    EngineCapability,
)
from aqua_ocr_smart_engine.vram_manager import VRAMManager, GLOBAL_VRAM_MANAGER
from aqua_ocr_smart_engine.preprocessor import DocumentPreprocessor, GLOBAL_PREPROCESSOR
from aqua_ocr_smart_engine.router import IntelligentRouter

from aqua_ocr_smart_engine.engines import (
    BaseOCREngine,
    PaddleOCREngine,
    TrOCREngine,
    IndicOCREngine,
    UnlimitedOCREngine,
    SuryaOCREngine,
    DocTREngine,
    MinerUEngine,
    TesseractEngine,
    HybridEnsembleEngine,
)

from aqua_ocr_smart_engine.postprocessor import (
    TableBuilder,
    ReadingOrderSorter,
    MarkdownBuilder,
)

__version__ = "1.0.0"
__author__ = "AQIVON Technologies"
__all__ = [
    "AquaOCRConfig",
    "OmniOCRConfig",
    "DEFAULT_CONFIG",
    "BoundingBox",
    "TextBlock",
    "TableCell",
    "TableResult",
    "PageResult",
    "OCRResult",
    "BlockType",
    "EngineCapability",
    "VRAMManager",
    "GLOBAL_VRAM_MANAGER",
    "DocumentPreprocessor",
    "GLOBAL_PREPROCESSOR",
    "IntelligentRouter",
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
    "TableBuilder",
    "ReadingOrderSorter",
    "MarkdownBuilder",
]
