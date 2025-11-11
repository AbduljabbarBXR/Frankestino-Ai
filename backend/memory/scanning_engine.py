"""
Continuous Semantic Scanning Engine
Background process for autonomous learning and relationship discovery
"""
import logging
import asyncio
import threading
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue

from .autonomous_mesh import AutonomousMesh
from ..utils.cache import get_cache

logger = logging.getLogger(__name__)


class ScanningConfiguration:
    """Configuration for semantic scanning behavior"""

    def __init__(self):
        self.scan_interval = 1.0  # seconds between scans
        self.similarity_threshold = 0.75
        self.max_scan_batch = 100  # neurons per batch
        self.max_workers = 4  # parallel scanning workers
        self.embedding_cache_ttl = 3600  # 1 hour
        self.scan_timeout = 30.0  # max time per scan iteration
        self.adaptive_threshold = True  # adjust threshold based on network size
        self.focus_recent = True  # prioritize recently activated neurons
        self.co_occurrence_weight = 0.3  # weight for co-occurrence patterns


class SemanticScanner:
    """
    Continuous semantic scanning engine that discovers relationships
    between neurons through embedding similarity and co-occurrence analysis
    """

    def __init__(self, autonomous_mesh: AutonomousMesh):
        self.mesh = autonomous_mesh
        self.config = ScanningConfiguration()

        # Scanning state
        self.is_scanning = False
        self.scan_thread = None
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # Queues for async processing
        self.scan_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Statistics
        self.scan_stats = {
            "total_scans": 0,
            "comparisons_made": 0,
            "relationships_found": 0,
            "connections_created": 0,
            "average_scan_time": 0.0,
            "last_scan_time": 0.0
        }

        # Caching
        self.cache = get_cache()
        self.similarity_cache = {}  # (neuron_a, neuron_b) -> similarity

        logger.info("Semantic Scanner initialized")

    def start_scanning(self):
        """Start the continuous scanning process"""
        if self.is_scanning:
            logger.warning("Scanning already active")
            return

        self.is_scanning = True
        self.scan_thread = threading.Thread(target=self._scanning_loop, daemon=True)
        self.scan_thread.start()

        logger.info("Started continuous semantic scanning")

    def stop_scanning(self):
        """Stop the scanning process"""
        self.is_scanning = False
        if self.scan_thread and self.scan_thread.is_alive():
            self.scan_thread.join(timeout=5.0)

        self.executor.shutdown(wait=True)
        logger.info("Stopped semantic scanning")

    def _scanning_loop(self):
        """Main scanning loop"""
        while self.is_scanning:
            try:
                start_time = time.time()

                # Perform semantic scan
                self._perform_scan_iteration()

                # Calculate scan time
                scan_time = time.time() - start_time
                self._update_scan_stats(scan_time)

                # Adaptive sleep based on scan time
                sleep_time = max(0.1, self.config.scan_interval - scan_time)
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in scanning loop: {e}")
                time.sleep(self.config.scan_interval)

    def _perform_scan_iteration(self):
        """Perform one iteration of semantic scanning"""
        if not self.mesh.nodes:
            return

        # Get neurons to scan
        neurons_to_scan = self._select_neurons_for_scanning()

        if len(neurons_to_scan) < 2:
            return

        # Perform similarity analysis
        similarity_pairs = self._calculate_similarity_matrix(neurons_to_scan)

        # Discover relationships
        relationships = self._discover_relationships(similarity_pairs)

        # Create autonomous connections
        connections_created = self._create_connections_from_relationships(relationships)

        # Update statistics
        self.scan_stats["comparisons_made"] += len(similarity_pairs)
        self.scan_stats["relationships_found"] += len(relationships)
        self.scan_stats["connections_created"] += connections_created

    def _select_neurons_for_scanning(self) -> List[str]:
        """Select neurons for this scanning iteration"""
        all_neurons = list(self.mesh.nodes.keys())

        # Adaptive batch size based on network size
        if self.config.adaptive_threshold:
            network_size = len(all_neurons)
            if network_size < 100:
                batch_size = min(network_size, 20)
            elif network_size < 1000:
                batch_size = min(network_size, 50)
            else:
                batch_size = self.config.max_scan_batch
        else:
            batch_size = self.config.max_scan_batch

        # Prioritize recently activated neurons if enabled
        if self.config.focus_recent:
            # Sort by activation level and recency
            sorted_neurons = sorted(
                all_neurons,
                key=lambda n: (
                    getattr(self.mesh.nodes[n], 'activation_level', 0),
                    getattr(self.mesh.nodes[n], 'last_accessed', 0)
                ),
                reverse=True
            )
            return sorted_neurons[:batch_size]
        else:
            # Random sampling
            if len(all_neurons) <= batch_size:
                return all_neurons
            return np.random.choice(all_neurons, batch_size, replace=False).tolist()

    def _calculate_similarity_matrix(self, neuron_ids: List[str]) -> List[Tuple[str, str, float]]:
        """Calculate similarity matrix for selected neurons"""
        similarity_pairs = []

        # Use thread pool for parallel similarity calculation
        futures = []

        for i, neuron_a_id in enumerate(neuron_ids):
            for neuron_b_id in neuron_ids[i+1:]:  # Upper triangle only
                future = self.executor.submit(
                    self._calculate_pair_similarity,
                    neuron_a_id,
                    neuron_b_id
                )
                futures.append((neuron_a_id, neuron_b_id, future))

        # Collect results
        for neuron_a_id, neuron_b_id, future in futures:
            try:
                similarity = future.result(timeout=10.0)
                if similarity >= 0.0:  # Valid similarity score
                    similarity_pairs.append((neuron_a_id, neuron_b_id, similarity))
            except Exception as e:
                logger.debug(f"Failed to calculate similarity for {neuron_a_id}-{neuron_b_id}: {e}")

        return similarity_pairs

    def _calculate_pair_similarity(self, neuron_a_id: str, neuron_b_id: str) -> float:
        """Calculate semantic similarity between two neurons"""
        # Check cache first
        cache_key = f"similarity_{neuron_a_id}_{neuron_b_id}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        neuron_a = self.mesh.nodes.get(neuron_a_id)
        neuron_b = self.mesh.nodes.get(neuron_b_id)

        if not neuron_a or not neuron_b:
            return -1.0

        # Get embeddings
        embedding_a = getattr(neuron_a, 'embedding', None)
        embedding_b = getattr(neuron_b, 'embedding', None)

        if not embedding_a or not embedding_b:
            return -1.0

        # Calculate cosine similarity
        similarity = self._cosine_similarity(embedding_a, embedding_b)

        # Add co-occurrence bonus if both are word neurons
        if (hasattr(neuron_a, 'word') and hasattr(neuron_b, 'word') and
            self.config.co_occurrence_weight > 0):
            co_occurrence = self._calculate_co_occurrence_bonus(neuron_a, neuron_b)
            similarity += co_occurrence * self.config.co_occurrence_weight
            similarity = min(1.0, similarity)  # Cap at 1.0

        # Cache the result
        self.cache.set(cache_key, similarity, ttl=self.config.embedding_cache_ttl)

        return similarity

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    def _calculate_co_occurrence_bonus(self, neuron_a, neuron_b) -> float:
        """Calculate co-occurrence bonus for word neurons"""
        if not (hasattr(neuron_a, 'contexts') and hasattr(neuron_b, 'contexts')):
            return 0.0

        contexts_a = set(neuron_a.contexts)
        contexts_b = set(neuron_b.contexts)

        if not contexts_a or not contexts_b:
            return 0.0

        # Calculate overlap coefficient
        intersection = len(contexts_a & contexts_b)
        min_size = min(len(contexts_a), len(contexts_b))

        if min_size == 0:
            return 0.0

        return intersection / min_size

    def _discover_relationships(self, similarity_pairs: List[Tuple[str, str, float]]) -> List[Dict[str, Any]]:
        """Discover semantic relationships from similarity data"""
        relationships = []

        # Adaptive threshold based on network size and current statistics
        threshold = self._calculate_adaptive_threshold()

        for neuron_a_id, neuron_b_id, similarity in similarity_pairs:
            if similarity >= threshold:
                relationship = self._analyze_relationship(neuron_a_id, neuron_b_id, similarity)
                if relationship:
                    relationships.append(relationship)

        return relationships

    def _calculate_adaptive_threshold(self) -> float:
        """Calculate adaptive similarity threshold"""
        if not self.config.adaptive_threshold:
            return self.config.similarity_threshold

        # Adjust threshold based on network size and recent activity
        network_size = len(self.mesh.nodes)
        connections_count = len(self.mesh.edges)

        # Lower threshold for smaller networks to encourage connections
        if network_size < 50:
            base_threshold = self.config.similarity_threshold * 0.8
        elif network_size < 200:
            base_threshold = self.config.similarity_threshold * 0.9
        else:
            base_threshold = self.config.similarity_threshold

        # Slightly lower threshold if few connections exist
        if connections_count < network_size * 0.1:  # Less than 10% connectivity
            base_threshold *= 0.95

        return min(base_threshold, 0.95)  # Never exceed 0.95

    def _analyze_relationship(self, neuron_a_id: str, neuron_b_id: str, similarity: float) -> Optional[Dict[str, Any]]:
        """Analyze the relationship between two neurons"""
        neuron_a = self.mesh.nodes.get(neuron_a_id)
        neuron_b = self.mesh.nodes.get(neuron_b_id)

        if not neuron_a or not neuron_b:
            return None

        # Determine relationship type
        relationship_type = self._classify_relationship(neuron_a, neuron_b, similarity)

        # Calculate confidence
        confidence = self._calculate_relationship_confidence(neuron_a, neuron_b, similarity)

        return {
            "neuron_a_id": neuron_a_id,
            "neuron_b_id": neuron_b_id,
            "similarity": similarity,
            "relationship_type": relationship_type,
            "confidence": confidence,
            "context": {
                "neuron_a_type": "word" if hasattr(neuron_a, 'word') else "memory",
                "neuron_b_type": "word" if hasattr(neuron_b, 'word') else "memory",
                "discovered_at": time.time()
            }
        }

    def _classify_relationship(self, neuron_a, neuron_b, similarity: float) -> str:
        """Classify the type of relationship between neurons"""
        # Word-to-word relationships
        if hasattr(neuron_a, 'word') and hasattr(neuron_b, 'word'):
            if similarity > 0.9:
                return "highly_similar"
            elif similarity > 0.8:
                return "similar"
            else:
                return "related"

        # Word-to-memory relationships
        elif hasattr(neuron_a, 'word') or hasattr(neuron_b, 'word'):
            return "contextual"

        # Memory-to-memory relationships
        else:
            return "associative"

    def _calculate_relationship_confidence(self, neuron_a, neuron_b, similarity: float) -> float:
        """Calculate confidence score for the relationship"""
        confidence = similarity  # Base confidence on similarity

        # Boost confidence for word neurons
        if hasattr(neuron_a, 'word') and hasattr(neuron_b, 'word'):
            confidence *= 1.1

        # Consider activation levels
        activation_a = getattr(neuron_a, 'activation_level', 0.0)
        activation_b = getattr(neuron_b, 'activation_level', 0.0)
        avg_activation = (activation_a + activation_b) / 2

        # Boost confidence for highly activated neurons
        if avg_activation > 0.5:
            confidence *= 1.05

        return min(confidence, 1.0)

    def _create_connections_from_relationships(self, relationships: List[Dict[str, Any]]) -> int:
        """Create autonomous connections from discovered relationships"""
        connections_created = 0

        for relationship in relationships:
            try:
                self.mesh._create_autonomous_connection(
                    relationship["neuron_a_id"],
                    relationship["neuron_b_id"],
                    relationship["similarity"]
                )
                connections_created += 1
            except Exception as e:
                logger.debug(f"Failed to create connection for relationship: {e}")

        return connections_created

    def get_scanning_stats(self) -> Dict[str, Any]:
        """Get current scanning statistics"""
        return {
            **self.scan_stats,
            "is_scanning": self.is_scanning,
            "config": {
                "scan_interval": self.config.scan_interval,
                "similarity_threshold": self.config.similarity_threshold,
                "max_scan_batch": self.config.max_scan_batch,
                "adaptive_threshold": self.config.adaptive_threshold
            }
        }

    def update_configuration(self, **kwargs):
        """Update scanning configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated scanning config {key} = {value}")

    def _update_scan_stats(self, scan_time: float):
        """Update scanning statistics"""
        self.scan_stats["total_scans"] += 1
        self.scan_stats["last_scan_time"] = scan_time

        # Update rolling average
        if self.scan_stats["total_scans"] == 1:
            self.scan_stats["average_scan_time"] = scan_time
        else:
            alpha = 0.1  # Smoothing factor
            self.scan_stats["average_scan_time"] = (
                alpha * scan_time +
                (1 - alpha) * self.scan_stats["average_scan_time"]
            )

    def __del__(self):
        """Cleanup"""
        self.stop_scanning()
