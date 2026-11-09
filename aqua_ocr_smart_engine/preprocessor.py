"""
Aqua OCR Smart Engine Document Preprocessor.
Handles PDF rendering, auto-deskewing, contrast adjustment, orientation correction,
and adaptive binarization for maximum OCR recognition accuracy.
"""

import io
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("aqua_ocr_smart_engine.preprocessor")


class DocumentPreprocessor:
    """Preprocesses images and multi-page PDFs for OCR engines."""

    def __init__(
        self,
        target_dpi: int = 200,
        max_dimension: int = 4096,
        enable_deskew: bool = True,
        enable_contrast_enhancement: bool = True
    ):
        self.target_dpi = target_dpi
        self.max_dimension = max_dimension
        self.enable_deskew = enable_deskew
        self.enable_contrast_enhancement = enable_contrast_enhancement
        
        # Check for local poppler binary if available on Windows
        self._poppler_path = self._detect_poppler_path()

    def _detect_poppler_path(self) -> Optional[str]:
        candidates = [
            r"G:\poppler\poppler-24.02.0\Library\bin",
            r"C:\Program Files\poppler\bin",
            r"C:\poppler\bin"
        ]
        for c in candidates:
            if os.path.exists(c) and os.path.exists(os.path.join(c, "pdftoppm.exe")):
                return c
        return None

    def load_document(self, input_source: Union[str, Path, bytes, Image.Image, np.ndarray]) -> List[np.ndarray]:
        """
        Loads document input into a list of RGB numpy images (one per page).
        Supports: file paths (image or PDF), raw bytes, PIL Image, or numpy array.
        """
        if isinstance(input_source, np.ndarray):
            # Single image numpy array
            if len(input_source.shape) == 2:
                rgb = cv2.cvtColor(input_source, cv2.COLOR_GRAY2RGB)
            elif input_source.shape[2] == 4:
                rgb = cv2.cvtColor(input_source, cv2.COLOR_RGBA2RGB)
            elif input_source.shape[2] == 3:
                rgb = input_source
            else:
                raise ValueError(f"Unsupported image shape: {input_source.shape}")
            return [self.preprocess_image(rgb)]

        if isinstance(input_source, Image.Image):
            rgb = np.array(input_source.convert("RGB"))
            return [self.preprocess_image(rgb)]

        if isinstance(input_source, bytes):
            # Check if PDF header
            if input_source.startswith(b"%PDF"):
                return self.render_pdf_bytes(input_source)
            # Otherwise decode image bytes
            nparr = np.frombuffer(input_source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes.")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return [self.preprocess_image(img_rgb)]

        # Assume file path
        path = Path(input_source)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self.render_pdf_file(path)
        else:
            # Load image from file using PIL (supports unicode paths reliably on Windows)
            with Image.open(path) as pil_img:
                rgb = np.array(pil_img.convert("RGB"))
            return [self.preprocess_image(rgb)]

    def render_pdf_file(self, pdf_path: Path) -> List[np.ndarray]:
        """Renders multi-page PDF into a list of RGB numpy arrays."""
        # Method 1: Try pypdfium2 (preferred: fast and self-contained)
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(str(pdf_path))
            pages = []
            scale = self.target_dpi / 72.0  # 72 dpi is 1.0x
            for page in pdf:
                pil_image = page.render(scale=scale).to_pil()
                rgb = np.array(pil_image.convert("RGB"))
                pages.append(self.preprocess_image(rgb))
            if pages:
                return pages
        except Exception as e:
            logger.debug("pypdfium2 rendering failed (%s). Falling back to pdf2image.", e)

        # Method 2: Try pdf2image with poppler
        try:
            from pdf2image import convert_from_path
            kwargs = {"dpi": self.target_dpi}
            if self._poppler_path:
                kwargs["poppler_path"] = self._poppler_path
            pil_images = convert_from_path(str(pdf_path), **kwargs)
            return [self.preprocess_image(np.array(img.convert("RGB"))) for img in pil_images]
        except Exception as e:
            logger.error("pdf2image rendering failed (%s)", e)
            raise RuntimeError(f"Failed to render PDF: {e}")

    def render_pdf_bytes(self, pdf_bytes: bytes) -> List[np.ndarray]:
        """Renders PDF bytes into a list of RGB numpy arrays."""
        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(pdf_bytes)
            pages = []
            scale = self.target_dpi / 72.0
            for page in pdf:
                pil_image = page.render(scale=scale).to_pil()
                rgb = np.array(pil_image.convert("RGB"))
                pages.append(self.preprocess_image(rgb))
            if pages:
                return pages
        except Exception as e:
            logger.debug("pypdfium2 bytes render failed (%s)", e)

        try:
            from pdf2image import convert_from_bytes
            kwargs = {"dpi": self.target_dpi}
            if self._poppler_path:
                kwargs["poppler_path"] = self._poppler_path
            pil_images = convert_from_bytes(pdf_bytes, **kwargs)
            return [self.preprocess_image(np.array(img.convert("RGB"))) for img in pil_images]
        except Exception as e:
            raise RuntimeError(f"Failed to render PDF bytes: {e}")

    def preprocess_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """Applies dimension normalization, deskewing, and contrast enhancements."""
        img = rgb_image.copy()

        # 1. Resize if exceeding maximum dimension to prevent VRAM spikes
        h, w = img.shape[:2]
        if max(h, w) > self.max_dimension:
            scale = self.max_dimension / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Deskew if enabled
        if self.enable_deskew:
            img = self.deskew(img)

        # 3. Enhance contrast if enabled
        if self.enable_contrast_enhancement:
            img = self.enhance_contrast(img)

        return img

    def deskew(self, rgb_image: np.ndarray) -> np.ndarray:
        """Detects skew angle and rotates the image back to upright."""
        try:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) < 100:
                return rgb_image  # Insufficient features to reliably estimate skew

            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle

            # Ignore negligible skew (< 0.5 deg) or extreme angles (> 45 deg)
            if abs(angle) < 0.5 or abs(angle) > 45.0:
                return rgb_image

            (h, w) = rgb_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                rgb_image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        except Exception as e:
            logger.debug("Deskew failed (%s); returning original image", e)
            return rgb_image

    def enhance_contrast(self, rgb_image: np.ndarray) -> np.ndarray:
        """Enhances contrast using CLAHE in LAB color space."""
        try:
            lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        except Exception:
            return rgb_image

    def to_binary(self, rgb_image: np.ndarray) -> np.ndarray:
        """Converts RGB image to optimized binary image for edge/text analysis."""
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )


# Global default preprocessor
GLOBAL_PREPROCESSOR = DocumentPreprocessor()

