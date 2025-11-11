"""
Neural Mesh Implementation
Dynamic graph structure with weighted connections for memory evolution
"""
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
from ..config import settings
from ..utils.mesh_accelerator import get_mesh_accelerator

logger = logging.getLogger(__name__)


@dataclass
class MeshNode:
    """A node in the neural mesh representing a memory chunk"""
    id: str
    content_hash: str  # Hash of the content for uniqueness
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    activation_level: float = 0.0  # Recent usage indicator
    created_at: float = 0.0
    last_accessed: float = 0.0

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at == 0.0:
            import time
            self.created_at = time.time()
        if self.last_accessed == 0.0:
            self.last_accessed = self.created_at


@dataclass
class MeshEdge:
    """A weighted connection between mesh nodes"""
    source_id: str
    target_id: str
    weight: float = 0.0  # Connection strength (0.0 to 1.0)
    connection_type: str = "similarity"  # similarity, usage, category
    created_at: float = 0.0
    last_reinforced: float = 0.0
    reinforcement_count: int = 0

    # NEW: Semantic relationship fields for Scaffolding & Substrate Model
    relationship_label: str = ""  # "is_capital_of", "contains", "provides", "is_located_in"
    relationship_type: str = ""   # "geographic", "causal", "definitional", "temporal", "associative"
    confidence: float = 0.5       # LLM confidence in this relationship (0.0 to 1.0)
    semantic_context: Optional[Dict[str, Any]] = None  # Additional relationship metadata

    def __post_init__(self):
        if self.created_at == 0.0:
            import time
            self.created_at = time.time()
        if self.last_reinforced == 0.0:
            self.last_reinforced = self.created_at
        if self.semantic_context is None:
            self.semantic_context = {}


class NeuralMesh:
    """
    Dynamic neural mesh for memory evolution
    Implements "digital neuroplasticity" through adaptive connections
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize neural mesh

        Args:
            storage_path: Path to save/load mesh data
        """
        self.storage_path = storage_path or settings.data_dir / "memory" / "neural_mesh.json"
        self.nodes: Dict[str, MeshNode] = {}
        self.edges: Dict[Tuple[str, str], MeshEdge] = {}
        self.adjacency_list: Dict[str, Set[str]] = defaultdict(set)

        # Mesh parameters
        self.max_connections_per_node = 10
        self.similarity_threshold = 0.7
        self.decay_rate = 0.95  # Daily decay factor
        self.reinforcement_boost = 0.1

        # Create storage directory
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing mesh
        self._load_mesh()

    def _load_mesh(self):
        """Load mesh from disk"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load nodes
                self.nodes = {}
                for node_data in data.get('nodes', []):
                    node = MeshNode(**node_data)
                    self.nodes[node.id] = node

                # Load edges
                self.edges = {}
                self.adjacency_list = defaultdict(set)
                for edge_data in data.get('edges', []):
                    edge = MeshEdge(**edge_data)
                    edge_key = (edge.source_id, edge.target_id)
                    self.edges[edge_key] = edge
                    self.adjacency_list[edge.source_id].add(edge.target_id)
                    self.adjacency_list[edge.target_id].add(edge.source_id)

                logger.info(f"Loaded neural mesh with {len(self.nodes)} nodes and {len(self.edges)} edges")

            except Exception as e:
                logger.warning(f"Failed to load neural mesh: {e}")
        else:
            logger.info("No existing neural mesh found, starting fresh")

    def _save_mesh(self):
        """Save mesh to disk"""
        try:
            data = {
                'nodes': [asdict(node) for node in self.nodes.values()],
                'edges': [asdict(edge) for edge in self.edges.values()],
                'metadata': {
                    'created_at': self._get_timestamp(),
                    'total_nodes': len(self.nodes),
                    'total_edges': len(self.edges)
                }
            }

            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug("Neural mesh saved to disk")
        except Exception as e:
            logger.error(f"Failed to save neural mesh: {e}")

    def _get_timestamp(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()

    def add_node(self, node_id: str, content_hash: str,
                 embedding: Optional[List[float]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a new node to the mesh

        Args:
            node_id: Unique node identifier
            content_hash: Hash of the content
            embedding: Vector embedding
            metadata: Additional metadata

        Returns:
            True if added, False if already exists
        """
        if node_id in self.nodes:
            return False

        node = MeshNode(
            id=node_id,
            content_hash=content_hash,
            embedding=embedding,
            metadata=metadata or {}
        )

        self.nodes[node_id] = node

        # Create connections to similar existing nodes
        self._create_similarity_connections(node)

        self._save_mesh()
        logger.debug(f"Added node {node_id} to neural mesh")
        return True

    def _create_similarity_connections(self, new_node: MeshNode):
        """Create connections based on embedding similarity"""
        if not new_node.embedding:
            return

        new_embedding = np.array(new_node.embedding)
        similarities = []

        # Calculate similarities with existing nodes
        for existing_node in self.nodes.values():
            if existing_node.id == new_node.id or not existing_node.embedding:
                continue

            existing_embedding = np.array(existing_node.embedding)
            similarity = self._cosine_similarity(new_embedding, existing_embedding)

            if similarity >= self.similarity_threshold:
                similarities.append((existing_node.id, similarity))

        # Sort by similarity and create top connections
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:self.max_connections_per_node]

        for similar_id, similarity in top_similar:
            self._add_edge(new_node.id, similar_id, similarity, "similarity")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _add_edge(self, source_id: str, target_id: str, weight: float,
                  connection_type: str = "similarity",
                  relationship_label: str = "",
                  relationship_type: str = "",
                  confidence: float = 0.5,
                  semantic_context: Optional[Dict[str, Any]] = None):
        """Add or update an edge between two nodes with semantic relationship support"""
        edge_key = (source_id, target_id)

        if edge_key in self.edges:
            # Update existing edge
            edge = self.edges[edge_key]
            edge.weight = min(1.0, edge.weight + weight * self.reinforcement_boost)
            edge.last_reinforced = self._get_timestamp()
            edge.reinforcement_count += 1

            # Update semantic fields if provided
            if relationship_label:
                edge.relationship_label = relationship_label
            if relationship_type:
                edge.relationship_type = relationship_type
            if confidence > edge.confidence:  # Only update if higher confidence
                edge.confidence = confidence
            if semantic_context:
                edge.semantic_context.update(semantic_context)
        else:
            # Create new edge
            edge = MeshEdge(
                source_id=source_id,
                target_id=target_id,
                weight=min(1.0, weight),
                connection_type=connection_type,
                relationship_label=relationship_label,
                relationship_type=relationship_type,
                confidence=confidence,
                semantic_context=semantic_context or {}
            )
            self.edges[edge_key] = edge

            # Update adjacency list
            self.adjacency_list[source_id].add(target_id)
            self.adjacency_list[target_id].add(source_id)

    def reinforce_connection(self, source_id: str, target_id: str,
                           reinforcement: float = 0.1):
        """
        Reinforce a connection (digital neuroplasticity)

        Args:
            source_id: Source node ID
            target_id: Target node ID
            reinforcement: Amount to increase connection strength
        """
        edge_key = (source_id, target_id)
        reverse_key = (target_id, source_id)

        # Try both directions
        if edge_key in self.edges:
            edge = self.edges[edge_key]
        elif reverse_key in self.edges:
            edge = self.edges[reverse_key]
        else:
            # Create new connection if it doesn't exist
            self._add_edge(source_id, target_id, reinforcement, "usage")
            return

        # Reinforce the connection
        edge.weight = min(1.0, edge.weight + reinforcement)
        edge.last_reinforced = self._get_timestamp()
        edge.reinforcement_count += 1

        # Update node activation levels
        if source_id in self.nodes:
            self.nodes[source_id].activation_level = min(1.0, self.nodes[source_id].activation_level + reinforcement * 0.5)
            self.nodes[source_id].last_accessed = self._get_timestamp()

        if target_id in self.nodes:
            self.nodes[target_id].activation_level = min(1.0, self.nodes[target_id].activation_level + reinforcement * 0.5)
            self.nodes[target_id].last_accessed = self._get_timestamp()

    def traverse_mesh(self, start_node_id: str, max_depth: int = 3,
                     min_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Traverse the mesh from a starting node using C++ acceleration when available

        Args:
            start_node_id: Starting node ID
            max_depth: Maximum traversal depth
            min_weight: Minimum edge weight to traverse

        Returns:
            List of reachable nodes with path information
        """
        if start_node_id not in self.nodes:
            return []

        # Try C++ accelerated traversal first
        accelerator = get_mesh_accelerator()
        try:
            accelerated_result = accelerator.accelerated_mesh_traversal(
                self.nodes, self.edges, start_node_id, max_depth, min_weight
            )

            if accelerated_result:
                # Convert C++ results to expected format
                result = []
                for node_id, weight in accelerated_result.items():
                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        result.append({
                            'node_id': node_id,
                            'depth': 0,  # C++ version doesn't track depth
                            'path': [],
                            'total_weight': weight,
                            'activation_level': node.activation_level,
                            'metadata': node.metadata
                        })
                logger.debug(f"Used C++ accelerated traversal: {len(result)} nodes")
                return result

        except Exception as e:
            logger.debug(f"C++ traversal failed, falling back to Python: {e}")

        # Fallback to Python implementation
        visited = set()
        result = []

        def dfs(current_id: str, depth: int, path: List[str], total_weight: float):
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)
            node = self.nodes[current_id]

            result.append({
                'node_id': current_id,
                'depth': depth,
                'path': path.copy(),
                'total_weight': total_weight,
                'activation_level': node.activation_level,
                'metadata': node.metadata
            })

            if depth < max_depth:
                # Get neighbors sorted by edge weight
                neighbors = []
                for neighbor_id in self.adjacency_list[current_id]:
                    edge_key = (current_id, neighbor_id)
                    reverse_key = (neighbor_id, current_id)

                    edge = self.edges.get(edge_key) or self.edges.get(reverse_key)
                    if edge and edge.weight >= min_weight:
                        neighbors.append((neighbor_id, edge.weight))

                # Sort by weight (highest first)
                neighbors.sort(key=lambda x: x[1], reverse=True)

                for neighbor_id, weight in neighbors[:5]:  # Limit branching
                    new_path = path + [current_id]
                    dfs(neighbor_id, depth + 1, new_path, total_weight + weight)

        dfs(start_node_id, 0, [], 0.0)
        logger.debug(f"Used Python traversal: {len(result)} nodes")
        return result

    def find_clusters(self, min_cluster_size: int = 3) -> List[List[str]]:
        """
        Find densely connected clusters in the mesh

        Args:
            min_cluster_size: Minimum cluster size

        Returns:
            List of node ID clusters
        """
        # Simple clustering based on connected components with high weights
        visited = set()
        clusters = []

        for node_id in self.nodes:
            if node_id not in visited:
                # Find connected component
                cluster = set()
                queue = [node_id]

                while queue:
                    current_id = queue.pop(0)
                    if current_id not in visited:
                        visited.add(current_id)
                        cluster.add(current_id)

                        # Add strongly connected neighbors
                        for neighbor_id in self.adjacency_list[current_id]:
                            edge_key = (current_id, neighbor_id)
                            reverse_key = (neighbor_id, current_id)

                            edge = self.edges.get(edge_key) or self.edges.get(reverse_key)
                            if edge and edge.weight >= 0.8:  # High weight threshold
                                if neighbor_id not in visited:
                                    queue.append(neighbor_id)

                if len(cluster) >= min_cluster_size:
                    clusters.append(list(cluster))

        return clusters

    def decay_weights(self, days_passed: float = 1.0):
        """
        Apply time-based decay to edge weights (forgetting mechanism)

        Args:
            days_passed: Number of days to decay
        """
        decay_factor = self.decay_rate ** days_passed

        decayed_count = 0
        for edge in self.edges.values():
            old_weight = edge.weight
            edge.weight *= decay_factor

            # Remove very weak connections
            if edge.weight < 0.05:
                edge.weight = 0.0

            if edge.weight != old_weight:
                decayed_count += 1

        # Also decay node activation levels
        for node in self.nodes.values():
            node.activation_level *= decay_factor

        if decayed_count > 0:
            logger.debug(f"Decayed {decayed_count} edge weights")
            self._save_mesh()

    def get_mesh_stats(self) -> Dict[str, Any]:
        """Get mesh statistics"""
        if not self.edges:
            avg_weight = 0.0
            max_weight = 0.0
        else:
            weights = [edge.weight for edge in self.edges.values()]
            avg_weight = sum(weights) / len(weights)
            max_weight = max(weights)

        # Calculate connectivity
        degrees = [len(self.adjacency_list[node_id]) for node_id in self.nodes.keys()]
        avg_degree = sum(degrees) / max(len(degrees), 1)

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "average_weight": avg_weight,
            "max_weight": max_weight,
            "average_degree": avg_degree,
            "clusters_found": len(self.find_clusters())
        }

    def cleanup_weak_connections(self, min_weight: float = 0.1):
        """
        Remove very weak connections to optimize memory

        Args:
            min_weight: Minimum weight to keep
        """
        edges_to_remove = []

        for edge_key, edge in self.edges.items():
            if edge.weight < min_weight:
                edges_to_remove.append(edge_key)

        for edge_key in edges_to_remove:
            edge = self.edges[edge_key]
            del self.edges[edge_key]

            # Update adjacency list
            self.adjacency_list[edge.source_id].discard(edge.target_id)
            self.adjacency_list[edge.target_id].discard(edge.source_id)

        if edges_to_remove:
            logger.info(f"Cleaned up {len(edges_to_remove)} weak connections")
            self._save_mesh()

    # ===== SCAFFOLDING & SUBSTRATE MODEL METHODS =====

    def add_semantic_edge(self, source_id: str, target_id: str,
                         relationship_label: str, relationship_type: str,
                         confidence: float = 0.5, weight: float = 0.5,
                         semantic_context: Optional[Dict[str, Any]] = None):
        """
        Add a semantic relationship edge between two nodes

        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_label: Semantic label (e.g., "is_capital_of")
            relationship_type: Relationship category (e.g., "geographic")
            confidence: LLM confidence score (0.0 to 1.0)
            weight: Initial connection weight
            semantic_context: Additional relationship metadata
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot create semantic edge: nodes {source_id} or {target_id} not found")
            return False

        self._add_edge(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            connection_type="semantic",
            relationship_label=relationship_label,
            relationship_type=relationship_type,
            confidence=confidence,
            semantic_context=semantic_context
        )

        logger.debug(f"Added semantic edge: {source_id} --[{relationship_label}]--> {target_id}")
        self._save_mesh()
        return True

    def reinforce_hebbian_connections(self, activated_nodes: List[str], reward: float = 0.1):
        """
        Hebbian learning: "Nodes that fire together, wire together"
        Strengthen connections between co-activated nodes

        Args:
            activated_nodes: List of node IDs that were activated together
            reward: Reinforcement amount
        """
        if len(activated_nodes) < 2:
            return

        reinforced_count = 0

        # Create connections between all pairs of activated nodes
        for i, node_a in enumerate(activated_nodes):
            for node_b in activated_nodes[i+1:]:
                if node_a != node_b:
                    self.reinforce_connection(node_a, node_b, reward)
                    reinforced_count += 1

        # Update activation levels for all nodes
        for node_id in activated_nodes:
            if node_id in self.nodes:
                self.nodes[node_id].activation_level = min(
                    1.0, self.nodes[node_id].activation_level + reward * 0.3
                )
                self.nodes[node_id].last_accessed = self._get_timestamp()

        logger.debug(f"Hebbian reinforcement: strengthened {reinforced_count} connections between {len(activated_nodes)} nodes")
        self._save_mesh()

    def get_semantic_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all semantic relationships for a node

        Args:
            node_id: Node ID to query

        Returns:
            List of semantic relationship dictionaries
        """
        relationships = []

        # Check outgoing edges
        for edge_key, edge in self.edges.items():
            if edge.source_id == node_id and edge.relationship_label:
                relationships.append({
                    'direction': 'outgoing',
                    'target_node': edge.target_id,
                    'relationship_label': edge.relationship_label,
                    'relationship_type': edge.relationship_type,
                    'weight': edge.weight,
                    'confidence': edge.confidence,
                    'semantic_context': edge.semantic_context
                })

        # Check incoming edges
        for edge_key, edge in self.edges.items():
            if edge.target_id == node_id and edge.relationship_label:
                relationships.append({
                    'direction': 'incoming',
                    'source_node': edge.source_id,
                    'relationship_label': edge.relationship_label,
                    'relationship_type': edge.relationship_type,
                    'weight': edge.weight,
                    'confidence': edge.confidence,
                    'semantic_context': edge.semantic_context
                })

        return relationships

    def find_related_nodes(self, node_id: str, relationship_type: str = None,
                          min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find nodes related to the given node through semantic relationships

        Args:
            node_id: Source node ID
            relationship_type: Filter by relationship type (optional)
            min_confidence: Minimum confidence threshold

        Returns:
            List of related nodes with relationship info
        """
        related = []

        for relationship in self.get_semantic_relationships(node_id):
            if relationship['confidence'] >= min_confidence:
                if relationship_type is None or relationship['relationship_type'] == relationship_type:
                    target_node_id = (relationship['target_node'] if relationship['direction'] == 'outgoing'
                                    else relationship['source_node'])

                    if target_node_id in self.nodes:
                        related.append({
                            'node_id': target_node_id,
                            'node_data': self.nodes[target_node_id],
                            'relationship': relationship
                        })

        return related

    def __len__(self) -> int:
        """Return number of nodes in mesh"""
        return len(self.nodes)
