"""
AQIVON OmniOCR Intelligent Engine Router.
Dynamically routes documents and images to the optimal OCR engine based on:
- Input language (Indian scripts, English, International)
- Document modality (Multi-page PDF, forms, handwritten notes, academic papers)
- RTX 3060 12GB VRAM constraints and GPU memory availability
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

from aqivon_omniocr.config import OmniOCRConfig, DEFAULT_CONFIG
from aqivon_omniocr.types import OCRResult, EngineCapability
from aqivon_omniocr.vram_manager import VRAMManager, GLOBAL_VRAM_MANAGER
from aqivon_omniocr.preprocessor import DocumentPreprocessor, GLOBAL_PREPROCESSOR

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

logger = logging.getLogger("aqivon_omniocr.router")


class IntelligentRouter:
    """
    Core orchestration router for AQIVON OmniOCR.
    Selects and invokes the best engine adapter for any given document.
    """

    def __init__(
        self,
        config: Optional[OmniOCRConfig] = None,
        vram_manager: Optional[VRAMManager] = None,
        preprocessor: Optional[DocumentPreprocessor] = None
    ):
        self.config = config or DEFAULT_CONFIG
        self.vram_manager = vram_manager or GLOBAL_VRAM_MANAGER
        self.preprocessor = preprocessor or GLOBAL_PREPROCESSOR

        # Instantiate all modular engine adapters
        self.engines: Dict[str, BaseOCREngine] = {
            "paddleocr": PaddleOCREngine(vram_manager=self.vram_manager, preprocessor=self.preprocessor),
            "indicocr": IndicOCREngine(vram_manager=self.vram_manager, preprocessor=self.preprocessor),
            "trocr_handwritten": TrOCREngine(mode="handwritten", vram_manager=self.vram_manager, preprocessor=self.preprocessor),
            "trocr_printed": TrOCREngine(mode="printed", vram_manager=self.vram_manager, preprocessor=self.preprocessor),
            "unlimited_ocr": UnlimitedOCREngine(vram_manager=self.vram_manager, preprocessor=self.preprocessor),
            "surya": SuryaOCREngine(vram_manager=self.vram_manager, preprocessor=self.preprocessor),
            "doctr": DocTREngine(vram_manager=self.vram_manager, preprocessor=self.preprocessor),
            "mineru": MinerUEngine(vram_manager=self.vram_manager, preprocessor=self.preprocessor),
            "tesseract": TesseractEngine(vram_manager=self.vram_manager, preprocessor=self.preprocessor),
            "hybrid": HybridEnsembleEngine(vram_manager=self.vram_manager, preprocessor=self.preprocessor),
        }

    def list_engines(self) -> List[EngineCapability]:
        """Returns capabilities of all registered engine adapters."""
        return [engine.get_capabilities() for engine in self.engines.values()]

    def route_and_process(
        self,
        document_source: Union[str, bytes, np.ndarray],
        requested_engine: Optional[str] = None,
        language: Optional[str] = "auto",
        is_handwritten: bool = False,
        document_name: str = "document"
    ) -> OCRResult:
        """
        Intelligently resolves optimal engine and runs end-to-end OCR.
        """
        selected_engine_id = self._select_engine(
            requested_engine=requested_engine,
            language=language,
            is_handwritten=is_handwritten
        )
        logger.info("Intelligent Router selected engine: '%s' for language: '%s'", selected_engine_id, language)

        engine = self.engines.get(selected_engine_id) or self.engines["hybrid"]

        # Run OCR
        result = engine.process_document(
            document_source=document_source,
            language=language,
            document_name=document_name
        )
        result.metadata["selected_by"] = "IntelligentRouter"
        result.metadata["requested_engine"] = requested_engine or "auto"
        return result

    def _select_engine(
        self,
        requested_engine: Optional[str],
        language: Optional[str],
        is_handwritten: bool
    ) -> str:
        """Heuristic decision tree for engine selection."""
        # 1. Explicit user override
        if requested_engine and requested_engine in self.engines:
            return requested_engine

        # 2. Handwriting priority
        if is_handwritten:
            return "trocr_handwritten"

        lang = (language or "auto").lower()

        # 3. Indian languages specialization
        if lang in IndicOCREngine.SUPPORTED_INDIC_LANGS:
            return "indicocr"

        # 4. Long documents / markdown layout priority
        if requested_engine == "unlimited_ocr":
            return "unlimited_ocr"

        # 5. Default high-precision ensemble
        return "hybrid"

