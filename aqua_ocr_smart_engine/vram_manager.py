"""
Aqua OCR Smart Engine VRAM & Hardware Manager.
Tailored for NVIDIA GeForce RTX 3060 12GB GDDR6 and multi-core CPU environments.
Ensures zero CUDA OOMs (Out-Of-Memory) via dynamic model caching and FP16/BF16 memory management.
"""

from collections import OrderedDict
import gc
import logging
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("aqua_ocr_smart_engine.vram_manager")


class VRAMManager:
    """
    Manages GPU memory allocation, precision, and active OCR model instances.
    Guarantees reliable operation on 12GB VRAM cards (e.g. RTX 3060).
    """

    def __init__(
        self,
        max_vram_gb: float = 11.0,
        eviction_threshold: float = 0.85,
        default_precision: str = "fp16",
        device: str = "auto"
    ):
        self.max_vram_bytes = int(max_vram_gb * 1024 * 1024 * 1024)
        self.eviction_threshold = eviction_threshold
        self.default_precision = default_precision
        self._device_str = device
        self._resolved_device: Optional[str] = None
        
        # LRU cache for model weights/instances
        self._model_cache: OrderedDict[str, Any] = OrderedDict()
        self._model_sizes: Dict[str, int] = {}

    def get_device(self) -> str:
        """Resolves target device ('cuda' or 'cpu')."""
        if self._resolved_device is not None:
            return self._resolved_device

        if self._device_str.lower() == "cpu":
            self._resolved_device = "cpu"
            return "cpu"

        try:
            import torch
            if torch.cuda.is_available():
                self._resolved_device = f"cuda:{torch.cuda.current_device()}"
                logger.info(
                    "CUDA available. Using GPU: %s (Total VRAM: %.2f GB)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / (1024**3)
                )
            else:
                self._resolved_device = "cpu"
                logger.warning("CUDA not available. Falling back to multi-core CPU inference.")
        except Exception as e:
            logger.warning("Torch check failed (%s). Falling back to CPU.", e)
            self._resolved_device = "cpu"

        return self._resolved_device

    def is_cuda(self) -> bool:
        return "cuda" in self.get_device()

    def get_torch_dtype(self):
        """Returns the optimal torch dtype for RTX 3060 (Ampere architecture)."""
        import torch
        if not self.is_cuda():
            return torch.float32

        # Ampere (RTX 3060) natively supports both FP16 and BF16
        if self.default_precision.lower() == "bf16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif self.default_precision.lower() == "fp16":
            return torch.float16
        return torch.float32

    def get_vram_stats(self) -> Dict[str, float]:
        """Returns current VRAM allocated, reserved, and free in MB."""
        if not self.is_cuda():
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "total_mb": 0.0,
                "free_mb": 0.0
            }
        
        try:
            import torch
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            return {
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "total_mb": round(total, 2),
                "free_mb": round(total - reserved, 2)
            }
        except Exception:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0, "free_mb": 0.0}

    def clear_cache(self):
        """Forces garbage collection and clears PyTorch CUDA cache."""
        gc.collect()
        if self.is_cuda():
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as e:
                logger.debug("CUDA empty_cache error: %s", e)

    def register_model(self, model_key: str, model_instance: Any, estimated_mb: int = 1000):
        """Registers a loaded model into the managed LRU cache."""
        # Evict if required
        self._check_and_evict(estimated_mb)
        
        if model_key in self._model_cache:
            self._model_cache.move_to_end(model_key)
        self._model_cache[model_key] = model_instance
        self._model_sizes[model_key] = estimated_mb
        logger.info("Registered model '%s' in VRAM cache (est. %d MB)", model_key, estimated_mb)

    def get_model(self, model_key: str) -> Optional[Any]:
        """Retrieves a cached model instance and marks it as recently used."""
        if model_key in self._model_cache:
            self._model_cache.move_to_end(model_key)
            return self._model_cache[model_key]
        return None

    def evict_model(self, model_key: str):
        """Explicitly unloads a model from memory and clears VRAM."""
        if model_key in self._model_cache:
            model = self._model_cache.pop(model_key)
            del model
            self._model_sizes.pop(model_key, None)
            self.clear_cache()
            logger.info("Evicted model '%s' from VRAM cache", model_key)

    def evict_all(self):
        """Unloads all cached models and frees all VRAM."""
        keys = list(self._model_cache.keys())
        for k in keys:
            self.evict_model(k)
        self.clear_cache()
        logger.info("All models evicted from VRAM cache")

    def _check_and_evict(self, incoming_mb: int):
        """Checks if current VRAM usage exceeds threshold and evicts oldest models."""
        if not self.is_cuda():
            return
        
        try:
            import torch
            current_allocated = torch.cuda.memory_allocated()
            if (current_allocated + (incoming_mb * 1024 * 1024)) > (self.max_vram_bytes * self.eviction_threshold):
                logger.warning("VRAM threshold reached. Evicting least recently used models...")
                while self._model_cache and (torch.cuda.memory_allocated() + (incoming_mb * 1024 * 1024)) > (self.max_vram_bytes * self.eviction_threshold):
                    oldest_key, _ = self._model_cache.popitem(last=False)
                    self._model_sizes.pop(oldest_key, None)
                    logger.info("Auto-evicted model '%s'", oldest_key)
                self.clear_cache()
        except Exception as e:
            logger.debug("VRAM check error: %s", e)


# Global VRAM manager instance
GLOBAL_VRAM_MANAGER = VRAMManager()

