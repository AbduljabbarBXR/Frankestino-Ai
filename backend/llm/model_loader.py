"""
GGUF Model Loader for Frankenstino AI
Handles loading and basic inference with llama.cpp
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from llama_cpp import Llama

from ..config import settings

logger = logging.getLogger(__name__)


class GGUFModelLoader:
    """Handles loading and basic operations for GGUF models"""

    def __init__(self, model_path: Optional[str] = None, model_type: str = "qwen"):
        """
        Initialize the model loader for unified Qwen model

        Args:
            model_path: Path to GGUF model file (Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf)
            model_type: Model type (qwen only - unified architecture)
        """
        self.model_type = model_type  # Set model_type first
        self.model_path = Path(model_path or self._get_default_model_path())
        self.model = None
        self.is_loaded = False

        # Model-specific parameters
        self.model_params = self._get_model_params()

        logger.info(f"Initialized GGUF model loader for: {self.model_path} (type: {model_type})")

    def _get_default_model_path(self) -> str:
        """Get the default model path based on model type"""
        if self.model_type == "deepseek":
            return settings.backend_model_path
        else:
            return settings.frontend_model_path

    def _get_model_params(self) -> Dict[str, Any]:
        """Get model parameters optimized for unified Qwen model"""
        base_params = {
            "n_ctx": settings.model_context_length,
            "n_threads": settings.model_threads,
            "n_batch": 512,
            "n_gpu_layers": 0,
            "verbose": False,
            "seed": 42,
            # MMAP: Enable memory mapping for efficient memory usage
            "use_mmap": True,
            # Don't lock pages in RAM - allow swapping
            "use_mlock": False,
        }

        # Unified Qwen model optimizations (single model architecture)
        if self.model_type == "qwen" or "qwen" in str(self.model_path):
            # Qwen2.5-7B-Instruct-1M model parameters
            base_params.update({
                "rope_freq_base": 1000000,  # Qwen-specific RoPE
                "rope_freq_scale": 1.0,
            })

        return base_params

    def load_model(self) -> bool:
        """
        Load the GGUF model

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.model_path.exists():
            logger.error(f"Model file not found: {self.model_path}")
            return False

        try:
            logger.info("Loading GGUF model... (this may take a moment)")

            self.model = Llama(
                model_path=str(self.model_path),
                **self.model_params
            )

            self.is_loaded = True
            logger.info("GGUF model loaded successfully")
            logger.info(f"Model info: {self.get_model_info()}")

            return True

        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            self.is_loaded = False
            return False

    def generate_text(self, prompt: str, max_tokens: int = 256,
                     temperature: float = 0.7, stop_sequences: Optional[List[str]] = None,
                     stream: bool = False) -> str:
        """
        Generate text from a prompt

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            stop_sequences: Sequences that stop generation
            stream: Whether to stream the response

        Returns:
            Generated text
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        # Default stop sequences for instruction-tuned models
        # Removed "\n\n" as it causes premature truncation of multi-paragraph responses
        if stop_sequences is None:
            stop_sequences = ["###", "User:", "Assistant:"]

        try:
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_sequences,
                echo=False,  # Don't include prompt in response
                stream=stream
            )

            if stream:
                return response  # Return generator for streaming
            else:
                return response["choices"][0]["text"]

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_loaded or self.model is None:
            return {"status": "not_loaded"}

        try:
            model_size = self.model_path.stat().st_size if self.model_path.exists() else 0
            return {
                "status": "loaded",
                "model_path": str(self.model_path),
                "context_length": self.model.n_ctx(),
                "vocab_size": self.model.n_vocab(),
                "embedding_dim": self.model.n_embd(),
                "threads": self.model_params["n_threads"],
                "model_size_mb": model_size / (1024 * 1024),
                "memory_mapping": {
                    "mmap_enabled": self.model_params.get("use_mmap", False),
                    "mlock_disabled": not self.model_params.get("use_mlock", True),
                    "memory_efficient": self.model_params.get("use_mmap", False)
                },
                "model_type": self.model_type
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"status": "error", "error": str(e)}

    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize text into token IDs"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        return self.model.tokenize(text.encode("utf-8"))

    def detokenize_text(self, tokens: List[int]) -> str:
        """Convert token IDs back to text"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        return self.model.detokenize(tokens).decode("utf-8")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        if not self.is_loaded or self.model is None:
            # Fallback estimation
            return len(text) // 4

        tokens = self.tokenize_text(text)
        return len(tokens)

    def unload_model(self):
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info("Model unloaded")

    def __del__(self):
        """Cleanup on deletion"""
        self.unload_model()
