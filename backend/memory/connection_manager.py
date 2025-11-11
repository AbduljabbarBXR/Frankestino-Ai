"""
Autonomous Connection Manager
Handles dynamic connection formation, management, and optimization
"""
import logging
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import heapq

from .autonomous_mesh import AutonomousMesh
from ..utils.cache import get_cache

logger = logging.getLogger(__name__)


class ConnectionConfig:
    """Configuration for connection management"""

    def __init__(self):
        self.max_connections_per_neuron = 50
        self.connection_decay_rate = 0.95  # daily decay factor
        self.min_connection_weight = 0.05
        self.pruning_interval = 3600  # seconds (1 hour)
        self.reinforcement_boost = 0.1
        self.connection_types = {
            "semantic": {"max_weight": 1.0, "decay_rate": 0.98},
            "syntactic": {"max_weight": 0.8, "decay_rate": 0.96},
            "associative": {"max_weight": 0.6, "decay_rate": 0.94},
            "analogical": {"max_weight": 0.7, "decay_rate": 0.97}
        }


class ConnectionManager:
    """
    Manages autonomous connection formation, reinforcement, and optimization
    in the word association network
    """

    def __init__(self, autonomous_mesh: AutonomousMesh):
        self.mesh = autonomous_mesh
        self.config = ConnectionConfig()

        # Connection tracking
        self.connection_history = defaultdict(list)  # neuron_id -> list of connection events
        self.pruning_schedule = {}  # neuron_id -> next_pruning_time

        # Statistics
        self.connection_stats = {
            "total_connections_created": 0,
            "connections_pruned": 0,
            "reinforcements_applied": 0,
            "average_connection_weight": 0.0,
            "connection_types_distribution": defaultdict(int)
        }

        # Caching
        self.cache = get_cache()

        logger.info("Connection Manager initialized")

    def create_connection(self, neuron_a_id: str, neuron_b_id: str,
                         similarity: float, connection_type: str = "autonomous",
                         relationship_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new connection between two neurons

        Args:
            neuron_a_id: First neuron ID
            neuron_b_id: Second neuron ID
            similarity: Semantic similarity score (0.0 to 1.0)
            connection_type: Type of connection
            relationship_context: Additional relationship metadata

        Returns:
            True if connection was created, False otherwise
        """
        if neuron_a_id == neuron_b_id:
            return False

        # Check if neurons exist
        if neuron_a_id not in self.mesh.nodes or neuron_b_id not in self.mesh.nodes:
            logger.warning(f"Cannot create connection: neuron not found")
            return False

        # Check connection limits
        if not self._can_create_connection(neuron_a_id, neuron_b_id):
            return False

        # Determine connection weight and type
        weight = self._calculate_connection_weight(similarity, connection_type)
        actual_connection_type = self._classify_connection_type(neuron_a_id, neuron_b_id, similarity)

        # Create semantic context
        semantic_context = relationship_context or {}
        semantic_context.update({
            "created_at": time.time(),
            "creation_method": "autonomous",
            "initial_similarity": similarity,
            "connection_type": actual_connection_type
        })

        # Add the edge
        self.mesh._add_edge(
            source_id=neuron_a_id,
            target_id=neuron_b_id,
            weight=weight,
            connection_type=actual_connection_type,
            relationship_label=self._generate_relationship_label(neuron_a_id, neuron_b_id, similarity),
            confidence=similarity,
            semantic_context=semantic_context
        )

        # Update statistics
        self.connection_stats["total_connections_created"] += 1
        self.connection_stats["connection_types_distribution"][actual_connection_type] += 1

        # Record in history
        self._record_connection_event(neuron_a_id, neuron_b_id, "created", weight)

        logger.debug(f"Created {actual_connection_type} connection between {neuron_a_id} and {neuron_b_id} (weight: {weight:.3f})")
        return True

    def _can_create_connection(self, neuron_a_id: str, neuron_b_id: str) -> bool:
        """Check if a connection can be created between two neurons"""
        # Check if connection already exists
        edge_key = (neuron_a_id, neuron_b_id)
        reverse_key = (neuron_b_id, neuron_a_id)

        if edge_key in self.mesh.edges or reverse_key in self.mesh.edges:
            return False

        # Check connection limits
        connections_a = len(self.mesh.adjacency_list[neuron_a_id])
        connections_b = len(self.mesh.adjacency_list[neuron_b_id])

        if (connections_a >= self.config.max_connections_per_neuron or
            connections_b >= self.config.max_connections_per_neuron):
            return False

        return True

    def _calculate_connection_weight(self, similarity: float, connection_type: str) -> float:
        """Calculate the initial weight for a connection"""
        # Base weight on similarity
        base_weight = similarity

        # Adjust based on connection type
        if connection_type in self.config.connection_types:
            max_weight = self.config.connection_types[connection_type]["max_weight"]
            base_weight = min(base_weight, max_weight)

        return base_weight

    def _classify_connection_type(self, neuron_a_id: str, neuron_b_id: str, similarity: float) -> str:
        """Classify the type of connection to create"""
        neuron_a = self.mesh.nodes.get(neuron_a_id)
        neuron_b = self.mesh.nodes.get(neuron_b_id)

        if not neuron_a or not neuron_b:
            return "associative"

        # Word-to-word connections
        if hasattr(neuron_a, 'word') and hasattr(neuron_b, 'word'):
            if similarity > 0.9:
                return "semantic"
            elif similarity > 0.8:
                return "syntactic"
            else:
                return "associative"

        # Word-to-memory connections
        elif hasattr(neuron_a, 'word') or hasattr(neuron_b, 'word'):
            return "contextual"

        # Memory-to-memory connections
        else:
            return "associative"

    def _generate_relationship_label(self, neuron_a_id: str, neuron_b_id: str, similarity: float) -> str:
        """Generate a human-readable relationship label"""
        neuron_a = self.mesh.nodes.get(neuron_a_id)
        neuron_b = self.mesh.nodes.get(neuron_b_id)

        if hasattr(neuron_a, 'word') and hasattr(neuron_b, 'word'):
            if similarity > 0.9:
                return "highly_similar"
            elif similarity > 0.8:
                return "similar"
            else:
                return "related"
        else:
            return "connected"

    def reinforce_connection(self, neuron_a_id: str, neuron_b_id: str,
                           reinforcement: float = None) -> bool:
        """
        Reinforce an existing connection (Hebbian learning)

        Args:
            neuron_a_id: First neuron ID
            neuron_b_id: Second neuron ID
            reinforcement: Reinforcement amount (uses config default if None)

        Returns:
            True if reinforcement was applied, False otherwise
        """
        if reinforcement is None:
            reinforcement = self.config.reinforcement_boost

        # Find the edge
        edge = self._find_edge(neuron_a_id, neuron_b_id)
        if not edge:
            return False

        # Apply reinforcement
        old_weight = edge.weight
        connection_type = edge.connection_type

        # Get max weight for this connection type
        max_weight = self.config.connection_types.get(connection_type, {}).get("max_weight", 1.0)

        # Apply reinforcement with diminishing returns
        edge.weight = min(max_weight, edge.weight + reinforcement)
        edge.last_reinforced = time.time()
        edge.reinforcement_count += 1

        # Update activation levels
        self._update_neuron_activation(neuron_a_id, reinforcement * 0.5)
        self._update_neuron_activation(neuron_b_id, reinforcement * 0.5)

        # Record reinforcement
        self._record_connection_event(neuron_a_id, neuron_b_id, "reinforced",
                                    edge.weight, old_weight=old_weight)

        # Update statistics
        self.connection_stats["reinforcements_applied"] += 1

        logger.debug(f"Reinforced connection {neuron_a_id}-{neuron_b_id}: {old_weight:.3f} -> {edge.weight:.3f}")
        return True

    def _find_edge(self, neuron_a_id: str, neuron_b_id: str):
        """Find an edge between two neurons"""
        edge_key = (neuron_a_id, neuron_b_id)
        reverse_key = (neuron_b_id, neuron_a_id)

        return (self.mesh.edges.get(edge_key) or
                self.mesh.edges.get(reverse_key))

    def _update_neuron_activation(self, neuron_id: str, activation_boost: float):
        """Update neuron activation level"""
        if neuron_id in self.mesh.nodes:
            neuron = self.mesh.nodes[neuron_id]
            neuron.activation_level = min(1.0, neuron.activation_level + activation_boost)
            neuron.last_accessed = time.time()

    def _record_connection_event(self, neuron_a_id: str, neuron_b_id: str,
                               event_type: str, new_weight: float,
                               old_weight: float = None):
        """Record a connection event in history"""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "neuron_a": neuron_a_id,
            "neuron_b": neuron_b_id,
            "new_weight": new_weight,
            "old_weight": old_weight
        }

        self.connection_history[neuron_a_id].append(event)
        self.connection_history[neuron_b_id].append(event)

        # Keep only recent history (last 100 events per neuron)
        for neuron_id in [neuron_a_id, neuron_b_id]:
            if len(self.connection_history[neuron_id]) > 100:
                self.connection_history[neuron_id] = self.connection_history[neuron_id][-100:]

    def apply_decay(self, days_passed: float = 1.0):
        """
        Apply time-based decay to connection weights

        Args:
            days_passed: Number of days to decay
        """
        decayed_count = 0
        total_weight = 0.0
        connection_count = 0

        for edge in self.mesh.edges.values():
            old_weight = edge.weight

            # Get decay rate for this connection type
            connection_type = edge.connection_type
            decay_rate = self.config.connection_types.get(connection_type, {}).get("decay_rate", self.config.connection_decay_rate)

            # Apply decay
            edge.weight *= (decay_rate ** days_passed)

            # Ensure minimum weight
            if edge.weight < self.config.min_connection_weight:
                edge.weight = 0.0

            if edge.weight != old_weight:
                decayed_count += 1

            if edge.weight > 0:
                total_weight += edge.weight
                connection_count += 1

        # Update average weight statistic
        if connection_count > 0:
            self.connection_stats["average_connection_weight"] = total_weight / connection_count

        if decayed_count > 0:
            logger.debug(f"Applied decay to {decayed_count} connections")

    def prune_weak_connections(self, min_weight: float = None) -> int:
        """
        Remove connections below minimum weight threshold

        Args:
            min_weight: Minimum weight to keep (uses config default if None)

        Returns:
            Number of connections pruned
        """
        if min_weight is None:
            min_weight = self.config.min_connection_weight

        edges_to_remove = []

        for edge_key, edge in self.mesh.edges.items():
            if edge.weight < min_weight:
                edges_to_remove.append(edge_key)

        # Remove the edges
        for edge_key in edges_to_remove:
            edge = self.mesh.edges[edge_key]

            # Update adjacency lists
            self.mesh.adjacency_list[edge.source_id].discard(edge.target_id)
            self.mesh.adjacency_list[edge.target_id].discard(edge.source_id)

            # Remove from mesh
            del self.mesh.edges[edge_key]

            # Record pruning event
            self._record_connection_event(edge.source_id, edge.target_id,
                                        "pruned", 0.0, old_weight=edge.weight)

        pruned_count = len(edges_to_remove)
        self.connection_stats["connections_pruned"] += pruned_count

        if pruned_count > 0:
            logger.info(f"Pruned {pruned_count} weak connections")

        return pruned_count

    def optimize_network(self) -> Dict[str, Any]:
        """
        Perform comprehensive network optimization

        Returns:
            Optimization results
        """
        results = {
            "connections_pruned": 0,
            "average_weight_before": self.connection_stats["average_connection_weight"],
            "total_connections_before": len(self.mesh.edges)
        }

        # Apply decay
        self.apply_decay(days_passed=1.0)

        # Prune weak connections
        results["connections_pruned"] = self.prune_weak_connections()

        # Update statistics
        results["average_weight_after"] = self.connection_stats["average_connection_weight"]
        results["total_connections_after"] = len(self.mesh.edges)

        logger.info(f"Network optimization completed: {results}")
        return results

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics"""
        # Calculate current metrics
        weights = [edge.weight for edge in self.mesh.edges.values()]
        avg_weight = sum(weights) / len(weights) if weights else 0.0

        # Connection type distribution
        type_dist = defaultdict(int)
        for edge in self.mesh.edges.values():
            type_dist[edge.connection_type] += 1

        # Degree distribution
        degrees = [len(self.mesh.adjacency_list[nid]) for nid in self.mesh.nodes.keys()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0

        return {
            **self.connection_stats,
            "current_average_weight": avg_weight,
            "total_active_connections": len(self.mesh.edges),
            "connection_type_distribution": dict(type_dist),
            "average_degree": avg_degree,
            "max_degree": max(degrees) if degrees else 0,
            "network_density": len(self.mesh.edges) / max(1, len(self.mesh.nodes) * (len(self.mesh.nodes) - 1) / 2)
        }

    def get_strongest_connections(self, neuron_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get the strongest connections for a neuron"""
        if neuron_id not in self.mesh.adjacency_list:
            return []

        connections = []
        for neighbor_id in self.mesh.adjacency_list[neuron_id]:
            edge = self._find_edge(neuron_id, neighbor_id)
            if edge:
                connections.append({
                    "neighbor_id": neighbor_id,
                    "weight": edge.weight,
                    "type": edge.connection_type,
                    "last_reinforced": edge.last_reinforced,
                    "reinforcement_count": edge.reinforcement_count
                })

        # Sort by weight descending
        connections.sort(key=lambda x: x["weight"], reverse=True)
        return connections[:top_k]

    def find_connection_path(self, start_id: str, end_id: str,
                           max_depth: int = 5) -> Optional[List[str]]:
        """
        Find a connection path between two neurons

        Args:
            start_id: Starting neuron ID
            end_id: Target neuron ID
            max_depth: Maximum path length

        Returns:
            List of neuron IDs representing the path, or None if no path found
        """
        if start_id not in self.mesh.nodes or end_id not in self.mesh.nodes:
            return None

        # BFS to find shortest path
        queue = [(start_id, [start_id])]
        visited = set([start_id])

        while queue:
            current_id, path = queue.pop(0)

            if current_id == end_id:
                return path

            if len(path) >= max_depth:
                continue

            for neighbor_id in self.mesh.adjacency_list[current_id]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def get_connection_recommendations(self, neuron_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend potential connections for a neuron based on network structure

        Args:
            neuron_id: Neuron to get recommendations for
            top_k: Number of recommendations to return

        Returns:
            List of recommended connections with scores
        """
        if neuron_id not in self.mesh.nodes:
            return []

        recommendations = []
        neuron = self.mesh.nodes[neuron_id]

        # Find neurons that are connected to this neuron's neighbors (triadic closure)
        neighbors = set(self.mesh.adjacency_list[neuron_id])
        potential_connections = set()

        for neighbor_id in neighbors:
            for second_neighbor_id in self.mesh.adjacency_list[neighbor_id]:
                if (second_neighbor_id != neuron_id and
                    second_neighbor_id not in neighbors):
                    potential_connections.add(second_neighbor_id)

        # Score potential connections
        for candidate_id in potential_connections:
            candidate = self.mesh.nodes.get(candidate_id)
            if not candidate:
                continue

            # Calculate recommendation score based on:
            # 1. Common neighbors
            # 2. Semantic similarity (if available)
            # 3. Network position

            common_neighbors = len(neighbors & set(self.mesh.adjacency_list[candidate_id]))
            score = common_neighbors * 0.1  # Base score from common neighbors

            # Add similarity bonus if both have embeddings
            if (hasattr(neuron, 'embedding') and neuron.embedding and
                hasattr(candidate, 'embedding') and candidate.embedding):
                try:
                    similarity = self._cosine_similarity(neuron.embedding, candidate.embedding)
                    score += similarity * 0.5
                except:
                    pass

            recommendations.append({
                "candidate_id": candidate_id,
                "score": score,
                "common_neighbors": common_neighbors,
                "candidate_type": "word" if hasattr(candidate, 'word') else "memory"
            })

        # Sort by score and return top_k
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vec1, dtype=np.float32)
            v2 = np.array(vec2, dtype=np.float32)

            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    def __del__(self):
        """Cleanup"""
        pass
