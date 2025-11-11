"""
Optimized Vector Store with FAISS quantization and advanced indexing
Supports IVFPQ, PQ, and HNSW indexing for scalable similarity search
"""

import os
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OptimizedVectorStore:
    """
    Advanced FAISS-based vector store with quantization and optimized indexing.
    Supports multiple indexing strategies for different performance/memory trade-offs.
    """

    def __init__(self, embedding_dim: int = 384, index_type: str = "flat",
                 nlist: int = 100, m: int = 8, nbits: int = 8,
                 use_gpu: bool = False, cache_dir: str = "data/embeddings"):
        """
        Initialize optimized vector store.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('ivf', 'ivfpq', 'pq', 'hnsw', 'flat')
            nlist: Number of clusters for IVF indexes
            m: Number of sub-quantizers for PQ
            nbits: Number of bits per sub-quantizer
            use_gpu: Whether to use GPU acceleration
            cache_dir: Directory for caching indexes
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.use_gpu = use_gpu
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Index and metadata storage
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}

        # Performance tracking
        self.query_count = 0
        self.index_size_mb = 0
        self.last_rebuild = None

        # Initialize index
        self._create_index()

    def _create_index(self):
        """Create the appropriate FAISS index based on configuration."""
        if self.index_type == "flat":
            # Exact search - highest accuracy, slowest
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        elif self.index_type == "ivf":
            # IVF with flat quantization - good balance
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)

        elif self.index_type == "ivfpq":
            # IVF with PQ - best memory/accuracy trade-off
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, self.nlist, self.m, self.nbits)

        elif self.index_type == "pq":
            # Pure PQ - lowest memory usage
            self.index = faiss.IndexPQ(self.embedding_dim, self.m, self.nbits)

        elif self.index_type == "hnsw":
            # HNSW - fastest search, higher memory
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)  # M=32

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Enable GPU if requested and available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
                logger.info("GPU acceleration enabled for FAISS index")
            except Exception as e:
                logger.warning(f"Failed to enable GPU acceleration: {e}")

        # Set search parameters for IVF indexes
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(10, self.nlist)  # Search more cells for better accuracy

        logger.info(f"Created {self.index_type} index with {self.embedding_dim} dimensions")

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]],
                   ids: Optional[List[str]] = None) -> List[str]:
        """
        Add vectors to the index with metadata.

        Args:
            vectors: Numpy array of shape (n, embedding_dim)
            metadata: List of metadata dictionaries
            ids: Optional list of IDs, generated if not provided

        Returns:
            List of assigned IDs
        """
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.embedding_dim}")

        n_vectors = len(vectors)

        # Generate IDs if not provided
        if ids is None:
            ids = [f"vec_{len(self.metadata) + i}" for i in range(n_vectors)]

        # Normalize vectors for cosine similarity (if using IP)
        try:
            # Check if index supports inner product (cosine similarity)
            if hasattr(self.index, 'metric_type') and self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                vectors = vectors / norms
        except (AttributeError, TypeError):
            # Fallback: assume IP metric for common index types
            if self.index_type in ['flat', 'ivf', 'ivfpq']:
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                vectors = vectors / norms

        # Train index if needed (for IVF and PQ)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if len(vectors) >= self.nlist * 39:  # Need enough vectors for training
                logger.info(f"Training {self.index_type} index with {len(vectors)} vectors")
                try:
                    self.index.train(vectors)
                except Exception as e:
                    logger.error(f"Failed to train {self.index_type} index: {e}")
                    # Fallback to IVF-Flat for reliability
                    if self.index_type in ['ivfpq', 'pq']:
                        logger.info(f"Falling back to IVF-Flat index due to training failure")
                        self._fallback_to_ivf_flat()
                        # Retry with fallback index
                        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                            self.index.train(vectors)
            else:
                logger.warning(f"Not enough vectors ({len(vectors)}) to train {self.index_type} index")
                # Fallback to IVF-Flat for small datasets
                if self.index_type in ['ivfpq', 'pq']:
                    logger.info(f"Falling back to IVF-Flat index (insufficient training data)")
                    self._fallback_to_ivf_flat()

        # Add vectors to index
        self.index.add(vectors.astype(np.float32))

        # Update metadata and ID mappings
        start_idx = len(self.metadata)
        for i, (id_val, meta) in enumerate(zip(ids, metadata)):
            idx = start_idx + i
            self.metadata.append(meta)
            self.id_to_idx[id_val] = idx
            self.idx_to_id[idx] = id_val

        # Update performance metrics
        self.index_size_mb = self._calculate_index_size_mb()
        self.last_rebuild = datetime.now()

        # Auto-upgrade index if we have enough data and are using flat
        total_vectors = len(self.metadata)
        if (self.index_type == "flat" and
            total_vectors >= self.nlist and
            total_vectors >= 50):  # Minimum threshold for meaningful IVF
            logger.info(f"Auto-upgrading index: Flat -> IVF (total vectors: {total_vectors})")
            self._upgrade_to_ivf()

        logger.info(f"Added {n_vectors} vectors to index. Total: {total_vectors}")
        return ids

    def search(self, query_vector: np.ndarray, top_k: int = 10,
              threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector of shape (embedding_dim,)
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (id, score, metadata) tuples
        """
        if query_vector.shape[0] != self.embedding_dim:
            raise ValueError(f"Query vector dimension {query_vector.shape[0]} doesn't match index dimension {self.embedding_dim}")

        # Handle empty index
        if self.index.ntotal == 0:
            logger.debug("Search attempted on empty index, returning empty results")
            self.query_count += 1
            return []

        # Normalize query vector
        try:
            # Check if index supports inner product (cosine similarity)
            if hasattr(self.index, 'metric_type') and self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = query_vector / norm
        except (AttributeError, TypeError):
            # Fallback: assume IP metric for common index types
            if self.index_type in ['flat', 'ivf', 'ivfpq']:
                norm = np.linalg.norm(query_vector)
                if norm > 0:
                    query_vector = query_vector / norm

        # Perform search - ensure k > 0
        actual_k = min(top_k, self.index.ntotal)
        if actual_k == 0:
            logger.debug("No vectors in index to search")
            self.query_count += 1
            return []

        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query_vector, actual_k)

        # Filter by threshold and format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:  # FAISS returns -1 for invalid results
                continue

            id_val = self.idx_to_id.get(idx)
            if id_val:
                metadata = self.metadata[idx].copy()
                metadata['index'] = idx
                results.append((id_val, float(score), metadata))

        self.query_count += 1

        # Log performance for first few queries
        if self.query_count <= 5:
            logger.info(f"Search query {self.query_count}: returned {len(results)} results, top_score={results[0][1] if results else 0}")

        return results[:top_k]

    def delete_vectors(self, ids: List[str]) -> int:
        """
        Delete vectors by IDs. Note: FAISS doesn't support deletion,
        so this rebuilds the index without the deleted vectors.

        Args:
            ids: List of vector IDs to delete

        Returns:
            Number of vectors deleted
        """
        ids_to_delete = set(ids)
        keep_indices = []
        keep_metadata = []
        keep_ids = []

        # Find indices to keep
        for idx, id_val in self.idx_to_id.items():
            if id_val not in ids_to_delete:
                keep_indices.append(idx)
                keep_metadata.append(self.metadata[idx])
                keep_ids.append(id_val)

        if not keep_indices:
            # No vectors left, create empty index
            self._create_index()
            self.metadata = []
            self.id_to_idx = {}
            self.idx_to_id = {}
            return len(ids)

        # Rebuild index with remaining vectors
        # This is inefficient but necessary since FAISS doesn't support deletion
        logger.info(f"Rebuilding index after deleting {len(ids)} vectors")

        # Extract remaining vectors (this assumes we can reconstruct them)
        # In practice, you'd want to store original vectors separately
        remaining_vectors = self._reconstruct_vectors(keep_indices)

        # Recreate index
        old_index_type = self.index_type
        self._create_index()

        # Re-add remaining vectors
        if len(remaining_vectors) > 0:
            self.add_vectors(remaining_vectors, keep_metadata, keep_ids)

        deleted_count = len(ids) - (len(keep_ids) - len(set(keep_ids) & ids_to_delete))
        logger.info(f"Deleted {deleted_count} vectors, {len(keep_ids)} remaining")

        return deleted_count

    def _reconstruct_vectors(self, indices: List[int]) -> np.ndarray:
        """
        Reconstruct vectors from index. Exact for flat indexes, approximate for quantized.
        In production, you'd store original vectors separately.
        """
        if self.index_type == "flat":
            # For flat indexes, we can reconstruct exactly by accessing stored vectors
            try:
                # FAISS flat index stores vectors directly, we can reconstruct them
                vectors = []
                for idx in indices:
                    if idx < self.index.ntotal:
                        vec = np.zeros(self.embedding_dim, dtype=np.float32)
                        self.index.reconstruct(idx, vec)
                        vectors.append(vec)
                    else:
                        logger.warning(f"Index {idx} out of bounds for reconstruction")
                return np.array(vectors) if vectors else np.array([])
            except Exception as e:
                logger.error(f"Failed to reconstruct vectors from flat index: {e}")
                return np.array([])

        elif hasattr(self.index, 'reconstruct'):
            # For other indexes that support reconstruction (like IVF)
            vectors = []
            for idx in indices:
                vec = np.zeros(self.embedding_dim, dtype=np.float32)
                self.index.reconstruct(idx, vec)
                vectors.append(vec)
            return np.array(vectors)
        else:
            # For indexes without reconstruction, we'd need to store vectors separately
            logger.warning("Vector reconstruction not supported for this index type")
            return np.array([])

    def save_index(self, filepath: str):
        """Save index and metadata to disk."""
        data = {
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'nlist': self.nlist,
            'm': self.m,
            'nbits': self.nbits,
            'metadata': self.metadata,
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id,
            'query_count': self.query_count,
            'last_rebuild': self.last_rebuild.isoformat() if self.last_rebuild else None
        }

        # Save metadata
        with open(f"{filepath}.meta", 'wb') as f:
            pickle.dump(data, f)

        # Save FAISS index
        if hasattr(self.index, 'cpu'):
            # Convert GPU index back to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index

        faiss.write_index(cpu_index, f"{filepath}.index")
        logger.info(f"Saved index to {filepath}")

    def load_index(self, filepath: str):
        """Load index and metadata from disk."""
        # Load metadata
        with open(f"{filepath}.meta", 'rb') as f:
            data = pickle.load(f)

        self.index_type = data['index_type']
        self.embedding_dim = data['embedding_dim']
        self.nlist = data['nlist']
        self.m = data['m']
        self.nbits = data['nbits']
        self.metadata = data['metadata']
        self.id_to_idx = data['id_to_idx']
        self.idx_to_id = data['idx_to_id']
        self.query_count = data['query_count']
        self.last_rebuild = datetime.fromisoformat(data['last_rebuild']) if data['last_rebuild'] else None

        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")

        # Re-enable GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            except Exception as e:
                logger.warning(f"Failed to load index to GPU: {e}")

        self.index_size_mb = self._calculate_index_size_mb()
        logger.info(f"Loaded index from {filepath} with {len(self.metadata)} vectors")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_vectors': len(self.metadata),
            'index_type': self.index_type,
            'embedding_dim': self.embedding_dim,
            'index_size_mb': self.index_size_mb,
            'query_count': self.query_count,
            'last_rebuild': self.last_rebuild.isoformat() if self.last_rebuild else None,
            'gpu_enabled': self.use_gpu and faiss.get_num_gpus() > 0
        }

    def optimize_for_memory(self):
        """Optimize index for memory efficiency."""
        if self.index_type == "flat":
            # Switch to IVF+PQ for memory savings
            logger.info("Optimizing index: Flat -> IVFPQ")
            self._rebuild_index("ivfpq")
        elif self.index_type == "ivf":
            # Switch to IVFPQ
            logger.info("Optimizing index: IVF -> IVFPQ")
            self._rebuild_index("ivfpq")

    def optimize_for_speed(self):
        """Optimize index for query speed."""
        if self.index_type in ["ivfpq", "pq"]:
            # Switch to IVF for speed
            logger.info("Optimizing index: PQ -> IVF")
            self._rebuild_index("ivf")
        elif self.index_type == "hnsw":
            # HNSW is already optimized for speed
            pass

    def _upgrade_to_ivf(self):
        """Upgrade from flat to IVF index when we have enough data."""
        try:
            # Store current data
            current_metadata = self.metadata.copy()
            current_ids = list(self.id_to_idx.keys())

            # For flat index, we need to reconstruct vectors from the index itself
            # This is exact for flat indexes
            indices = list(range(len(self.metadata)))
            vectors = self._reconstruct_vectors(indices)

            if len(vectors) == 0:
                logger.warning("Cannot upgrade to IVF: no vectors available")
                return

            # Switch to IVF
            old_type = self.index_type
            self.index_type = "ivf"

            # Create new IVF index
            self._create_index()

            # Add all vectors (this will trigger training since IVF needs it)
            self.add_vectors(vectors, current_metadata, current_ids)

            logger.info(f"Successfully upgraded index: {old_type} -> {self.index_type}")

        except Exception as e:
            logger.error(f"Failed to upgrade to IVF index: {e}")
            # Revert to flat on failure
            self.index_type = "flat"
            self._create_index()

    def _rebuild_index(self, new_index_type: str):
        """Rebuild index with new type."""
        # Store current vectors and metadata
        current_metadata = self.metadata.copy()
        current_ids = list(self.id_to_idx.keys())

        # Reconstruct vectors
        indices = list(range(len(self.metadata)))
        vectors = self._reconstruct_vectors(indices)

        if len(vectors) == 0:
            logger.warning("Cannot rebuild index: no vectors to reconstruct")
            return

        # Create new index
        old_type = self.index_type
        self.index_type = new_index_type
        self._create_index()

        # Re-add all vectors
        self.add_vectors(vectors, current_metadata, current_ids)
        logger.info(f"Rebuilt index: {old_type} -> {new_index_type}")

    def _calculate_index_size_mb(self) -> float:
        """Calculate approximate index size in MB."""
        if self.index is None:
            return 0

        # Rough estimation based on index type
        n_vectors = len(self.metadata)
        if self.index_type == "flat":
            # 4 bytes per float * dimensions * vectors
            size_bytes = 4 * self.embedding_dim * n_vectors
        elif self.index_type == "ivfpq":
            # Much smaller due to quantization
            size_bytes = n_vectors * (self.m * self.nbits / 8) + (self.nlist * self.embedding_dim * 4)
        else:
            # Conservative estimate
            size_bytes = n_vectors * self.embedding_dim * 2

        return size_bytes / (1024 * 1024)

    def _fallback_to_ivf_flat(self):
        """Fallback to IVF-Flat index when PQ training fails."""
        logger.info(f"Falling back from {self.index_type} to IVF-Flat")
        self.index_type = "ivf"
        # Recreate index with IVF-Flat
        self._create_index()

    def clear_cache(self):
        """Clear any cached data."""
        # Implementation for cache clearing
        pass
