"""
C++ Acceleration Hooks for Neural Mesh Operations
Provides high-performance implementations for mesh traversal and computation
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import logging
import subprocess
import importlib.util

logger = logging.getLogger(__name__)

class MeshAccelerator:
    """
    C++ accelerated neural mesh operations for performance-critical computations.
    Falls back to Python implementations if C++ extension is not available.
    """

    def __init__(self):
        self.cpp_available = False
        self.cpp_module = None
        self._initialize_cpp_extension()

    def _initialize_cpp_extension(self):
        """Try to load the C++ extension module."""
        try:
            # Try to import the C++ extension
            import mesh_accelerator_cpp
            self.cpp_module = mesh_accelerator_cpp
            self.cpp_available = True
            logger.info("C++ mesh accelerator loaded successfully")
        except ImportError:
            logger.warning("C++ mesh accelerator not available, using Python fallback")
            self.cpp_available = False
            self._build_cpp_extension()

    def _build_cpp_extension(self):
        """Attempt to build the C++ extension if source is available."""
        try:
            extension_dir = Path(__file__).parent / "cpp_extensions"
            if extension_dir.exists():
                logger.info("Found C++ extension source, attempting to build...")
                # This would trigger a build process in a real implementation
                # For now, we'll just log the attempt
                pass
        except Exception as e:
            logger.debug(f"C++ extension build failed: {e}")

    def accelerated_mesh_traversal(self, nodes: Dict[str, Any],
                                 edges: List[Tuple[str, str, float]],
                                 start_node: str, max_depth: int = 3,
                                 min_weight: float = 0.1) -> Dict[str, float]:
        """
        Perform accelerated mesh traversal using C++ implementation.

        Args:
            nodes: Dictionary of node_id -> node_data
            edges: List of (from_node, to_node, weight) tuples
            start_node: Starting node for traversal
            max_depth: Maximum traversal depth
            min_weight: Minimum edge weight to traverse

        Returns:
            Dictionary of reachable node_id -> path_weight
        """
        if self.cpp_available and self.cpp_module:
            try:
                # Convert to C++ compatible format
                node_ids = list(nodes.keys())
                node_index = {node_id: i for i, node_id in enumerate(node_ids)}

                # Build adjacency list for C++
                adj_list = [[] for _ in node_ids]
                edge_weights = []

                for from_node, to_node, weight in edges:
                    if from_node in node_index and to_node in node_index and weight >= min_weight:
                        from_idx = node_index[from_node]
                        to_idx = node_index[to_node]
                        adj_list[from_idx].append(to_idx)
                        edge_weights.append(weight)

                # Call C++ function
                start_idx = node_index.get(start_node, -1)
                if start_idx == -1:
                    return {}

                result_indices, result_weights = self.cpp_module.traverse_mesh(
                    adj_list, edge_weights, start_idx, max_depth
                )

                # Convert back to node IDs
                results = {}
                for idx, weight in zip(result_indices, result_weights):
                    if idx < len(node_ids):
                        results[node_ids[idx]] = weight

                return results

            except Exception as e:
                logger.warning(f"C++ traversal failed, falling back to Python: {e}")
                return self._python_mesh_traversal(nodes, edges, start_node, max_depth, min_weight)
        else:
            return self._python_mesh_traversal(nodes, edges, start_node, max_depth, min_weight)

    def accelerated_weighted_pagerank(self, nodes: Dict[str, Any],
                                     edges: List[Tuple[str, str, float]],
                                     damping_factor: float = 0.85,
                                     max_iterations: int = 100,
                                     tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Compute weighted PageRank using C++ acceleration.

        Args:
            nodes: Dictionary of node_id -> node_data
            edges: List of (from_node, to_node, weight) tuples
            damping_factor: PageRank damping factor
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            Dictionary of node_id -> pagerank_score
        """
        if self.cpp_available and self.cpp_module:
            try:
                # Convert to C++ format
                node_ids = list(nodes.keys())
                node_index = {node_id: i for i, node_id in enumerate(node_ids)}

                # Build weighted adjacency matrix
                n_nodes = len(node_ids)
                adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)

                for from_node, to_node, weight in edges:
                    if from_node in node_index and to_node in node_index:
                        from_idx = node_index[from_node]
                        to_idx = node_index[to_node]
                        adj_matrix[from_idx, to_idx] = weight

                # Call C++ PageRank
                pagerank_scores = self.cpp_module.weighted_pagerank(
                    adj_matrix, damping_factor, max_iterations, tolerance
                )

                # Convert back to dictionary
                results = {}
                for i, score in enumerate(pagerank_scores):
                    results[node_ids[i]] = score

                return results

            except Exception as e:
                logger.warning(f"C++ PageRank failed, falling back to Python: {e}")
                return self._python_weighted_pagerank(nodes, edges, damping_factor, max_iterations, tolerance)
        else:
            return self._python_weighted_pagerank(nodes, edges, damping_factor, max_iterations, tolerance)

    def accelerated_similarity_matrix(self, vectors: np.ndarray,
                                    similarity_type: str = "cosine",
                                    chunk_size: int = 1000) -> np.ndarray:
        """
        Compute similarity matrix using C++ acceleration for large datasets.

        Args:
            vectors: Numpy array of shape (n_vectors, n_features)
            similarity_type: Type of similarity ('cosine', 'euclidean', 'dot')
            chunk_size: Chunk size for memory-efficient computation

        Returns:
            Similarity matrix of shape (n_vectors, n_vectors)
        """
        if self.cpp_available and self.cpp_module:
            try:
                # Call C++ similarity computation
                similarity_matrix = self.cpp_module.compute_similarity_matrix(
                    vectors.astype(np.float32), similarity_type, chunk_size
                )
                return similarity_matrix

            except Exception as e:
                logger.warning(f"C++ similarity computation failed, falling back to Python: {e}")
                return self._python_similarity_matrix(vectors, similarity_type)
        else:
            return self._python_similarity_matrix(vectors, similarity_type)

    def accelerated_cluster_embeddings(self, vectors: np.ndarray,
                                     n_clusters: int,
                                     method: str = "kmeans") -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform clustering on embeddings using C++ acceleration.

        Args:
            vectors: Numpy array of shape (n_vectors, n_features)
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'hierarchical')

        Returns:
            Tuple of (cluster_labels, cluster_centers)
        """
        if self.cpp_available and self.cpp_module:
            try:
                labels, centers = self.cpp_module.cluster_embeddings(
                    vectors.astype(np.float32), n_clusters, method
                )
                return labels, centers

            except Exception as e:
                logger.warning(f"C++ clustering failed, falling back to Python: {e}")
                return self._python_cluster_embeddings(vectors, n_clusters, method)
        else:
            return self._python_cluster_embeddings(vectors, n_clusters, method)

    # Python fallback implementations

    def _python_mesh_traversal(self, nodes: Dict[str, Any],
                             edges: List[Tuple[str, str, float]],
                             start_node: str, max_depth: int = 3,
                             min_weight: float = 0.1) -> Dict[str, float]:
        """Python implementation of mesh traversal."""
        if start_node not in nodes:
            return {}

        # Build adjacency list
        adj_list = {}
        for from_node, to_node, weight in edges:
            if weight >= min_weight:
                if from_node not in adj_list:
                    adj_list[from_node] = []
                adj_list[from_node].append((to_node, weight))

        # BFS traversal with weights
        visited = set()
        queue = [(start_node, 0.0, 0)]  # (node, path_weight, depth)
        results = {start_node: 1.0}  # Start node has weight 1.0

        while queue:
            current_node, path_weight, depth = queue.pop(0)

            if current_node in visited or depth >= max_depth:
                continue

            visited.add(current_node)

            if current_node in adj_list:
                for neighbor, edge_weight in adj_list[current_node]:
                    if neighbor not in visited:
                        new_weight = path_weight + edge_weight
                        if neighbor not in results or new_weight > results[neighbor]:
                            results[neighbor] = new_weight
                            queue.append((neighbor, new_weight, depth + 1))

        return results

    def _python_weighted_pagerank(self, nodes: Dict[str, Any],
                                edges: List[Tuple[str, str, float]],
                                damping_factor: float = 0.85,
                                max_iterations: int = 100,
                                tolerance: float = 1e-6) -> Dict[str, float]:
        """Python implementation of weighted PageRank."""
        node_ids = list(nodes.keys())
        n_nodes = len(node_ids)
        node_index = {node_id: i for i, node_id in enumerate(node_ids)}

        # Build weighted adjacency matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for from_node, to_node, weight in edges:
            if from_node in node_index and to_node in node_index:
                from_idx = node_index[from_node]
                to_idx = node_index[to_node]
                adj_matrix[to_idx, from_idx] = weight  # Note: incoming edges

        # Normalize columns (make stochastic)
        column_sums = adj_matrix.sum(axis=0)
        column_sums[column_sums == 0] = 1  # Avoid division by zero
        adj_matrix = adj_matrix / column_sums

        # Initialize PageRank
        pagerank = np.ones(n_nodes) / n_nodes

        # Power iteration
        for iteration in range(max_iterations):
            new_pagerank = (1 - damping_factor) / n_nodes + damping_factor * adj_matrix.dot(pagerank)

            # Check convergence
            if np.linalg.norm(new_pagerank - pagerank, 1) < tolerance:
                break

            pagerank = new_pagerank

        # Convert to dictionary
        results = {}
        for i, score in enumerate(pagerank):
            results[node_ids[i]] = score

        return results

    def _python_similarity_matrix(self, vectors: np.ndarray,
                                similarity_type: str = "cosine") -> np.ndarray:
        """Python implementation of similarity matrix computation."""
        if similarity_type == "cosine":
            # Cosine similarity
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = vectors / norms
            return np.dot(normalized, normalized.T)

        elif similarity_type == "euclidean":
            # Convert to similarity (1 / (1 + distance))
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(vectors, 'euclidean'))
            return 1 / (1 + distances)

        elif similarity_type == "dot":
            return np.dot(vectors, vectors.T)

        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")

    def _python_cluster_embeddings(self, vectors: np.ndarray,
                                 n_clusters: int,
                                 method: str = "kmeans") -> Tuple[np.ndarray, np.ndarray]:
        """Python implementation of embedding clustering."""
        if method == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            centers = kmeans.cluster_centers_
            return labels, centers

        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the accelerator."""
        return {
            'cpp_available': self.cpp_available,
            'cpp_module': self.cpp_module.__name__ if self.cpp_module else None,
            'supported_operations': [
                'mesh_traversal',
                'weighted_pagerank',
                'similarity_matrix',
                'cluster_embeddings'
            ] if self.cpp_available else []
        }

# Global accelerator instance
_accelerator_instance = None

def get_mesh_accelerator() -> MeshAccelerator:
    """Get global mesh accelerator instance."""
    global _accelerator_instance
    if _accelerator_instance is None:
        _accelerator_instance = MeshAccelerator()
    return _accelerator_instance

# Convenience functions for easy access
def traverse_mesh(nodes: Dict[str, Any], edges: List[Tuple[str, str, float]],
                 start_node: str, max_depth: int = 3, min_weight: float = 0.1) -> Dict[str, float]:
    """Convenience function for mesh traversal."""
    accelerator = get_mesh_accelerator()
    return accelerator.accelerated_mesh_traversal(nodes, edges, start_node, max_depth, min_weight)

def compute_weighted_pagerank(nodes: Dict[str, Any], edges: List[Tuple[str, str, float]],
                             damping_factor: float = 0.85, max_iterations: int = 100,
                             tolerance: float = 1e-6) -> Dict[str, float]:
    """Convenience function for weighted PageRank."""
    accelerator = get_mesh_accelerator()
    return accelerator.accelerated_weighted_pagerank(nodes, edges, damping_factor, max_iterations, tolerance)

def compute_similarity_matrix(vectors: np.ndarray, similarity_type: str = "cosine",
                            chunk_size: int = 1000) -> np.ndarray:
    """Convenience function for similarity matrix computation."""
    accelerator = get_mesh_accelerator()
    return accelerator.accelerated_similarity_matrix(vectors, similarity_type, chunk_size)

def cluster_embeddings(vectors: np.ndarray, n_clusters: int,
                      method: str = "kmeans") -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function for embedding clustering."""
    accelerator = get_mesh_accelerator()
    return accelerator.accelerated_cluster_embeddings(vectors, n_clusters, method)
