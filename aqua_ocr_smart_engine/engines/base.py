"""
Aqua OCR Smart Engine Base Engine Specification.
Defines the abstract interface that every modular OCR adapter must implement.
"""

from abc import ABC, abstractmethod
import logging
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np

from aqua_ocr_smart_engine.types import (
    BoundingBox,
    TextBlock,
    PageResult,
    OCRResult,
    EngineCapability,
)
from aqua_ocr_smart_engine.preprocessor import DocumentPreprocessor, GLOBAL_PREPROCESSOR
from aqua_ocr_smart_engine.vram_manager import VRAMManager, GLOBAL_VRAM_MANAGER
from aqua_ocr_smart_engine.postprocessor.markdown_builder import MarkdownBuilder
from aqua_ocr_smart_engine.postprocessor.reading_order import ReadingOrderSorter

logger = logging.getLogger("aqua_ocr_smart_engine.engines.base")


class BaseOCREngine(ABC):
    """Abstract base class for all Aqua OCR Smart Engine adapters."""

    def __init__(
        self,
        engine_id: str,
        display_name: str,
        vram_manager: Optional[VRAMManager] = None,
        preprocessor: Optional[DocumentPreprocessor] = None
    ):
        self.engine_id = engine_id
        self.display_name = display_name
        self.vram_manager = vram_manager or GLOBAL_VRAM_MANAGER
        self.preprocessor = preprocessor or GLOBAL_PREPROCESSOR
        self._is_initialized = False

    @abstractmethod
    def get_capabilities(self) -> EngineCapability:
        """Returns metadata describing what this engine can handle."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks if the engine's dependencies and models can run in the current environment."""
        pass

    @abstractmethod
    def initialize(self):
        """Pre-loads models and prepares inference pipelines."""
        pass

    @abstractmethod
    def process_image(self, image: np.ndarray, language: Optional[str] = None) -> PageResult:
        """
        Executes OCR on a single preprocessed RGB image.
        Returns a PageResult containing text blocks, tables, and page-level markdown.
        """
        pass

    def process_document(
        self,
        document_source: Union[str, bytes, np.ndarray],
        language: Optional[str] = None,
        document_name: str = "document"
    ) -> OCRResult:
        """
        Executes full OCR pipeline over a multi-page document or image.
        Handles loading, page-by-page inference, reading-order structuring, and Markdown generation.
        """
        if not self._is_initialized:
            self.initialize()

        start_time = time.time()
        pages_images = self.preprocessor.load_document(document_source)
        page_results: List[PageResult] = []

        for page_idx, img in enumerate(pages_images, start=1):
            page_start = time.time()
            page_res = self.process_image(img, language=language)
            page_res.page_number = page_idx
            page_res.latency_ms = (time.time() - page_start) * 1000.0

            # Ensure reading order sorting
            page_res.blocks = ReadingOrderSorter.sort_blocks(
                page_res.blocks, page_res.width, page_res.height
            )

            # Rebuild clean page markdown if empty
            if not page_res.markdown:
                page_res.markdown = MarkdownBuilder.blocks_to_markdown(
                    page_res.blocks, page_res.tables
                )

            # Reconstruct layout-faithful HTML preview
            from aqua_ocr_smart_engine.postprocessor.html_reconstructor import HTMLDocumentReconstructor
            page_res.html_preview = HTMLDocumentReconstructor.render_page(page_res)
            
            # Extract plain text
            page_res.text = "\n".join(b.text for b in page_res.blocks if b.text.strip())
            page_results.append(page_res)

        total_latency_ms = (time.time() - start_time) * 1000.0
        vram_stats = self.vram_manager.get_vram_stats()

        # Build full document text, markdown, and layout HTML
        full_text_parts = [f"--- Page {p.page_number} ---\n{p.text}" for p in page_results]
        full_markdown_parts = [f"\n\n<!-- Page {p.page_number} -->\n{p.markdown}" for p in page_results]
        full_html_parts = [p.html_preview for p in page_results]

        result = OCRResult(
            document_name=document_name,
            num_pages=len(page_results),
            pages=page_results,
            full_text="\n\n".join(full_text_parts).strip(),
            markdown="\n\n".join(full_markdown_parts).strip(),
            html_preview="\n<hr style='margin: 40px 0; border: none; border-top: 2px dashed #94a3b8;' />\n".join(full_html_parts),
            engine_used=self.engine_id,
            detected_language=language or "auto",
            total_latency_ms=total_latency_ms,
            vram_usage_mb=vram_stats.get("allocated_mb"),
            metadata={
                "engine_name": self.display_name,
                "device": self.vram_manager.get_device(),
                "num_pages": len(page_results),
                "total_blocks": sum(len(p.blocks) for p in page_results)
            }
        )
        return result

