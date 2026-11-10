"""
Aqua OCR Smart Engine Intelligent Engine Router.
Dynamically routes documents and images to the optimal OCR engine based on:
- Input language (Indian scripts, English, International)
- Document modality (Multi-page PDF, forms, handwritten notes, academic papers)
- RTX 3060 12GB VRAM constraints and GPU memory availability
"""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np

from aqua_ocr_smart_engine.config import AquaOCRConfig, DEFAULT_CONFIG
from aqua_ocr_smart_engine.types import OCRResult, EngineCapability
from aqua_ocr_smart_engine.vram_manager import VRAMManager, GLOBAL_VRAM_MANAGER
from aqua_ocr_smart_engine.preprocessor import DocumentPreprocessor, GLOBAL_PREPROCESSOR

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

logger = logging.getLogger("aqua_ocr_smart_engine.router")


class IntelligentRouter:
    """
    Core orchestration router for Aqua OCR Smart Engine.
    Selects and invokes the best engine adapter for any given document.
    """

    def __init__(
        self,
        config: Optional[AquaOCRConfig] = None,
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

        # 5. Default fast & robust multilingual OCR
        return "paddleocr"

