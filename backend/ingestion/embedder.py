"""
Embedding Generation for Documents
CPU-optimized embedding generation using sentence-transformers
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import hashlib
from collections import OrderedDict
from ..config import settings

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Handles document embedding generation"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder

        Args:
            model_name: SentenceTransformer model name
        """
        self.model_name = model_name
        self.model = None
        self.embedding_cache = OrderedDict()  # LRU cache for embeddings
        self.cache_max_size = settings.embedding_cache_max_size  # Limit from config
        self._load_model()

    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Optimize for CPU
            self.model.to('cpu')

            logger.info(f"Model loaded successfully. Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode_texts(self, texts: List[str], batch_size: int = None,
                    show_progress_bar: bool = True, use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts with optional caching

        Args:
            texts: List of text strings
            batch_size: Batch size for processing (uses config default if None)
            show_progress_bar: Whether to show progress bar
            use_cache: Whether to use embedding cache

        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        batch_size = batch_size or settings.batch_size

        # Check cache for existing embeddings
        if use_cache:
            cached_embeddings = []
            texts_to_encode = []
            cache_indices = []

            for i, text in enumerate(texts):
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                if text_hash in self.embedding_cache:
                    # Move to end (most recently used)
                    self.embedding_cache.move_to_end(text_hash)
                    cached_embeddings.append((i, self.embedding_cache[text_hash]))
                else:
                    texts_to_encode.append(text)
                    cache_indices.append(i)

            # Encode missing texts
            if texts_to_encode:
                logger.info(f"Generating embeddings for {len(texts_to_encode)} new texts (cached: {len(cached_embeddings)})")

                try:
                    new_embeddings = self.model.encode(
                        texts_to_encode,
                        batch_size=batch_size,
                        show_progress_bar=show_progress_bar,
                        convert_to_numpy=True
                    )

                    # Cache new embeddings with LRU eviction
                    for text, embedding in zip(texts_to_encode, new_embeddings):
                        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                        self._add_to_cache(text_hash, embedding)

                    logger.info(f"Generated and cached embeddings with shape: {new_embeddings.shape}")
                except Exception as e:
                    logger.error(f"Failed to generate embeddings: {e}")
                    raise
            else:
                new_embeddings = np.array([])

            # Combine cached and new embeddings in correct order
            result_embeddings = np.zeros((len(texts), self.get_dimension()))
            for i, cached_emb in cached_embeddings:
                result_embeddings[i] = cached_emb

            if len(new_embeddings) > 0:
                for i, (cache_idx, new_emb) in enumerate(zip(cache_indices, new_embeddings)):
                    result_embeddings[cache_idx] = new_emb

            return result_embeddings

        else:
            # No caching - direct encoding
            logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")

            try:
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=True
                )

                logger.info(f"Generated embeddings with shape: {embeddings.shape}")
                return embeddings

            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                raise

    def encode_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks

        Args:
            chunks: List of chunk dictionaries from DocumentProcessor

        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            return chunks

        # Extract texts
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings
        embeddings = self.encode_texts(texts)

        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()  # Convert to list for JSON serialization

        logger.info(f"Added embeddings to {len(chunks)} chunks")
        return chunks

    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        if self.model is None:
            return settings.embedding_dim
        return self.model.get_sentence_embedding_dimension()

    def _add_to_cache(self, key: str, value: np.ndarray):
        """
        Add item to LRU cache with size limit enforcement

        Args:
            key: Cache key (text hash)
            value: Embedding vector
        """
        # Remove if already exists (will be re-added at end)
        if key in self.embedding_cache:
            del self.embedding_cache[key]

        # Add to end (most recently used)
        self.embedding_cache[key] = value

        # Evict least recently used items if over limit
        while len(self.embedding_cache) > self.cache_max_size:
            # Remove oldest (first) item
            oldest_key, _ = self.embedding_cache.popitem(last=False)
            logger.debug(f"Evicted embedding from cache: {oldest_key[:8]}...")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.embedding_cache),
            'max_size': self.cache_max_size,
            'utilization_percent': (len(self.embedding_cache) / self.cache_max_size) * 100 if self.cache_max_size > 0 else 0
        }

    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")

    def __call__(self, texts: List[str]) -> np.ndarray:
        """Convenience method for encoding texts"""
        return self.encode_texts(texts)
