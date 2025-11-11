"""
Configuration management for Frankenstino AI
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Project paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "models")

    # UNIFIED MODEL: Single Qwen model for all tasks (frontend + backend)
    unified_model_path: str = Field(default="backend/models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf")

    # Legacy compatibility (deprecated - use unified path)
    frontend_model_path: str = Field(default="backend/models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf")  # Point to unified
    backend_model_path: str = Field(default="backend/models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf")   # Point to unified
    model_path: str = Field(default="backend/models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf")           # Point to unified
    curator_model_path: str = Field(default="backend/models/Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf")   # Point to unified

    model_context_length: int = Field(default=32768)  # Qwen/DeepSeek have large context
    model_threads: int = Field(default=4)

    # Memory settings
    embedding_dim: int = Field(default=384)  # Reduced for efficiency
    max_chunk_size: int = Field(default=1024)  # Reduced for semantic chunking
    chunk_overlap: int = Field(default=50)
    semantic_chunking: bool = Field(default=True)  # Enable semantic chunking

    # Neural connectivity settings
    connectivity_strategy: str = Field(default="sliding_window")  # sliding_window, syntax_aware, attention_based, full
    connectivity_window_size: int = Field(default=3)  # Window size for sliding window connectivity
    connectivity_min_weight: float = Field(default=0.1)  # Minimum connection weight
    enable_syntax_parsing: bool = Field(default=False)  # Enable spaCy syntax parsing (requires installation)

    # Vector DB settings
    faiss_index_type: str = Field(default="flat")  # Flat index - exact search, works with any number of vectors
    nlist: int = Field(default=100)  # Number of clusters for IVF
    use_gpu_faiss: bool = Field(default=True)  # Enable GPU acceleration for FAISS
    gpu_memory_limit_mb: int = Field(default=6144)  # 6GB limit (safe for 7.7GB GPU)

    # Cache settings
    cache_size_mb: int = Field(default=500)
    cache_ttl_hours: int = Field(default=24)
    embedding_cache_max_size: int = Field(default=10000)  # Limit embedding cache size

    # API settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    cors_origins: list = Field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8000", "null", "file://"])

    # Performance settings
    max_workers: int = Field(default=4)
    batch_size: int = Field(default=32)

    # ===== SCAFFOLDING & SUBSTRATE MODEL - Phase 2: Signal Propagation System =====
    # Signal propagation settings
    enable_signal_propagation: bool = Field(default=True)
    signal_decay_rate: float = Field(default=0.7)  # How much activation decays per hop
    activation_threshold: float = Field(default=0.1)  # Minimum activation to propagate
    max_propagation_steps: int = Field(default=5)  # Maximum propagation steps

    # Concept extraction settings
    enable_concept_extraction: bool = Field(default=True)  # Use LLM for concept extraction

    # Pattern completion settings
    pattern_coherence_threshold: float = Field(default=0.6)  # Minimum coherence for valid patterns
    min_cluster_size: int = Field(default=3)  # Minimum nodes in a cluster
    max_answer_nodes: int = Field(default=10)  # Maximum answer nodes to return

    # Substrate autonomy settings (Phase 3 preparation)
    autonomy_maturity_threshold: float = Field(default=0.80)  # Maturity score for full autonomy
    gradual_transition_enabled: bool = Field(default=True)  # Enable gradual LLM dependency reduction
    substrate_only_simple_queries: bool = Field(default=False)  # Enable substrate-only for simple queries

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
