"""
AQIVON OmniOCR
A unified, production-ready AI OCR engine by AQIVON Technologies.
"""

from aqivon_omniocr.config import OmniOCRConfig, DEFAULT_CONFIG
from aqivon_omniocr.types import (
    BoundingBox,
    TextBlock,
    TableCell,
    TableResult,
    PageResult,
    OCRResult,
    BlockType,
    EngineCapability,
)
from aqivon_omniocr.vram_manager import VRAMManager, GLOBAL_VRAM_MANAGER
from aqivon_omniocr.preprocessor import DocumentPreprocessor, GLOBAL_PREPROCESSOR
from aqivon_omniocr.router import IntelligentRouter

from aqivon_omniocr.engines import (
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

from aqivon_omniocr.postprocessor import (
    TableBuilder,
    ReadingOrderSorter,
    MarkdownBuilder,
)

__version__ = "1.0.0"
__author__ = "AQIVON Technologies"
__all__ = [
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
