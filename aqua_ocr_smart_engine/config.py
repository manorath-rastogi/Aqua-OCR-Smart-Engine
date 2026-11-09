"""
Aqua OCR Smart Engine Configuration Module.
Engine configuration, hardware detection, and RTX 3060 12GB optimizations.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AquaOCRConfig:
    """Master configuration for Aqua OCR Smart Engine."""
    
    # Engine & Branding
    system_name: str = "Aqua OCR Smart Engine"
    organization: str = "AQIVON Technologies"
    version: str = "1.0.0"
    
    # Device & Hardware Settings
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    cuda_device_id: int = 0
    mixed_precision: str = "fp16"  # 'fp16', 'bf16', 'fp32'
    max_gpu_memory_gb: float = 11.0  # RTX 3060 has 12GB VRAM; leave 1GB headroom for OS/Display
    vram_eviction_threshold: float = 0.85  # Trigger eviction when 85% of allocated VRAM is reached
    
    # Document Preprocessing
    pdf_render_dpi: int = 200  # Optimal trade-off between OCR accuracy and processing latency
    max_image_dimension: int = 4096
    enable_deskew: bool = True
    enable_contrast_enhancement: bool = True
    
    # Supported Languages
    supported_languages: List[str] = field(default_factory=lambda: [
        "auto",
        "en",        # English
        "hi",        # Hindi (हिन्दी)
        "mr",        # Marathi (मराठी)
        "ur",        # Urdu (اردو)
        "bn",        # Bengali (বাংলা)
        "ta",        # Tamil (தமிழ்)
        "te",        # Telugu (తెలుగు)
        "gu",        # Gujarati (ગુજરાતી)
        "kn",        # Kannada (ಕನ್ನಡ)
        "ml",        # Malayalam (മലയാളം)
        "pa",        # Punjabi (ਪੰਜਾਬੀ)
        "or",        # Odia (ଓଡ଼ିଆ)
        "zh",        # Chinese (Simplified)
        "ar",        # Arabic
        "ru",        # Russian
        "es",        # Spanish
        "fr",        # French
        "de",        # German
        "ja",        # Japanese
    ])
    
    # Language Code to Readable Name
    language_display_names: Dict[str, str] = field(default_factory=lambda: {
        "auto": "Auto-Detect Language",
        "en": "English",
        "hi": "Hindi (हिन्दी)",
        "mr": "Marathi (मराठी)",
        "ur": "Urdu (اردو)",
        "bn": "Bengali (বাংলা)",
        "ta": "Tamil (தமிழ்)",
        "te": "Telugu (తెలుగు)",
        "gu": "Gujarati (ગુજરાતી)",
        "kn": "Kannada (ಕನ್ನಡ)",
        "ml": "Malayalam (മലയാളം)",
        "pa": "Punjabi (ਪੰਜਾਬੀ)",
        "or": "Odia (ଓଡ଼ିଆ)",
        "zh": "Chinese (Simplified)",
        "ar": "Arabic (العربية)",
        "ru": "Russian (Русский)",
        "es": "Spanish (Español)",
        "fr": "French (Français)",
        "de": "German (Deutsch)",
        "ja": "Japanese (日本語)",
    })
    
    # Storage and Cache
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("AQUA_CACHE_DIR", Path.home() / ".cache" / "aqua_ocr_smart_engine")))
    temp_dir: Path = field(default_factory=lambda: Path(os.getenv("AQUA_TEMP_DIR", Path.home() / ".cache" / "aqua_ocr_smart_engine" / "temp")))
    
    # Default Engine Routing Strategy
    default_engine: str = "intelligent_router"  # 'intelligent_router', 'unlimited_ocr', 'paddleocr', 'trocr', 'indicocr', 'surya', 'doctr', 'tesseract', 'hybrid'

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


OmniOCRConfig = AquaOCRConfig
DEFAULT_CONFIG = AquaOCRConfig()
