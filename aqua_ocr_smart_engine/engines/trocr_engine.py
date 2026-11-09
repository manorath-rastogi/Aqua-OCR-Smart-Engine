"""
Aqua OCR Smart Engine TrOCR Adapter.
Transformer-based OCR (VisionEncoderDecoder) specialized for Handwritten Text Recognition (HTR)
and degraded document line recognition using Microsoft TrOCR architectures.
Optimized for NVIDIA RTX 3060 12GB via FP16 tensor acceleration.
"""

import logging
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from aqua_ocr_smart_engine.engines.base import BaseOCREngine
from aqua_ocr_smart_engine.types import (
    BoundingBox,
    TextBlock,
    BlockType,
    PageResult,
    EngineCapability,
)

logger = logging.getLogger("aqua_ocr_smart_engine.engines.trocr_engine")


class TrOCREngine(BaseOCREngine):
    """
    Adapter for Microsoft TrOCR Transformer models.
    Excels at cursive handwriting, doctor prescriptions, signatures, and non-standard typography.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        printed_model_name: str = "microsoft/trocr-base-printed",
        mode: str = "handwritten",  # 'handwritten' or 'printed'
        **kwargs
    ):
        super().__init__(
            engine_id="trocr",
            display_name=f"TrOCR Transformer Engine ({mode.capitalize()})",
            **kwargs
        )
        self.selected_model_name = model_name if mode == "handwritten" else printed_model_name
        self.mode = mode
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None

    def get_capabilities(self) -> EngineCapability:
        return EngineCapability(
            engine_id=self.engine_id,
            display_name=self.display_name,
            supports_handwriting=True,
            supports_tables=False,
            supports_multipage_pdf=True,
            supports_layout_analysis=False,
            supported_languages=["en", "la", "numbers"],
            gpu_accelerated=self.vram_manager.is_cuda(),
            typical_vram_requirement_mb=1800.0,
            description="State-of-the-art Transformer encoder-decoder for cursive and handwritten text recognition."
        )

    def is_available(self) -> bool:
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            return True
        except ImportError:
            return False

    def initialize(self):
        """Loads TrOCR weights into PyTorch with FP16 precision."""
        if not self.is_available():
            raise RuntimeError("transformers and PyTorch are required for TrOCR.")

        cached = self.vram_manager.get_model(f"trocr_{self.selected_model_name}")
        if cached:
            self.processor, self.model = cached
            self._is_initialized = True
            return

        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        device = self.vram_manager.get_device()
        dtype = self.vram_manager.get_torch_dtype()

        logger.info("Loading TrOCR model '%s' onto %s...", self.selected_model_name, device)
        try:
            self.processor = TrOCRProcessor.from_pretrained(self.selected_model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.selected_model_name,
                torch_dtype=dtype if self.vram_manager.is_cuda() else torch.float32
            )
            self.model.to(device)
            self.model.eval()

            # Cache model
            self.vram_manager.register_model(
                f"trocr_{self.selected_model_name}",
                (self.processor, self.model),
                estimated_mb=1600
            )
            self._is_initialized = True
        except Exception as e:
            logger.error("Failed to load TrOCR model '%s': %s", self.selected_model_name, e)
            raise

    def process_image(self, image: np.ndarray, language: Optional[str] = None) -> PageResult:
        """
        Extracts lines from document image and recognizes handwriting via TrOCR.
        """
        if not self._is_initialized:
            self.initialize()

        h, w = image.shape[:2]
        
        # Segment image into line candidate crops
        line_crops_with_bbox = self._segment_lines(image)
        if not line_crops_with_bbox:
            return PageResult(page_number=1, width=w, height=h, blocks=[])

        import torch
        device = self.vram_manager.get_device()
        blocks: List[TextBlock] = []

        # Batch inference over line crops to maximize RTX 3060 GPU throughput
        batch_size = 8 if self.vram_manager.is_cuda() else 2
        for i in range(0, len(line_crops_with_bbox), batch_size):
            batch = line_crops_with_bbox[i:i + batch_size]
            pil_images = [Image.fromarray(b_img) for (b_img, _) in batch]

            try:
                pixel_values = self.processor(pil_images, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)
                if self.vram_manager.is_cuda() and self.vram_manager.default_precision == "fp16":
                    pixel_values = pixel_values.half()

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_new_tokens=64,
                        early_stopping=True,
                        no_repeat_ngram_size=3
                    )
                generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

                for (b_img, bbox), text in zip(batch, generated_texts):
                    cleaned_text = text.strip()
                    if cleaned_text:
                        blocks.append(TextBlock(
                            text=cleaned_text,
                            bbox=bbox,
                            confidence=0.92,
                            language="en",
                            block_type=BlockType.HANDWRITTEN if self.mode == "handwritten" else BlockType.PARAGRAPH,
                            is_handwritten=(self.mode == "handwritten"),
                            font_size_estimate=float(bbox.height)
                        ))
            except Exception as e:
                logger.warning("TrOCR batch inference error: %s", e)

        return PageResult(
            page_number=1,
            width=w,
            height=h,
            blocks=blocks,
            tables=[]
        )

    def _segment_lines(self, rgb_image: np.ndarray) -> List[Tuple[np.ndarray, BoundingBox]]:
        """
        Fast morphological line segmentation to slice a document into line crops.
        """
        h, w = rgb_image.shape[:2]
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Otsu binarization
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Horizontal dilation kernel to connect characters within the same text line
        kernel_width = max(15, int(w * 0.03))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        line_crops: List[Tuple[np.ndarray, BoundingBox]] = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # Filter noise
            if cw < 20 or ch < 10 or (cw * ch) < 250:
                continue
            
            # Add small padding
            pad_y = int(ch * 0.15)
            pad_x = int(cw * 0.03)
            y1 = max(0, y - pad_y)
            y2 = min(h, y + ch + pad_y)
            x1 = max(0, x - pad_x)
            x2 = min(w, x + cw + pad_x)

            crop = rgb_image[y1:y2, x1:x2]
            bbox = BoundingBox(xmin=float(x1), ymin=float(y1), xmax=float(x2), ymax=float(y2))
            line_crops.append((crop, bbox))

        # Sort top-to-bottom
        line_crops.sort(key=lambda item: item[1].ymin)
        return line_crops

