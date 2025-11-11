"""
Signal Processor - Query Activation and Pattern Completion Engine
Part of Scaffolding & Substrate Model - Phase 2
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

from ..config import settings

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Converts natural language queries to network activation patterns
    and finds answers through signal propagation and pattern completion.

    Part of Phase 2: Signal Propagation System
    """

    def __init__(self, neural_mesh, memory_manager, llm_interface=None):
        """
        Initialize signal processor

        Args:
            neural_mesh: NeuralMesh instance
            memory_manager: MemoryManager instance
            llm_interface: Optional LLM interface for concept extraction
        """
        self.neural_mesh = neural_mesh
        self.memory_manager = memory_manager
        self.llm_interface = llm_interface

        # Signal propagation parameters
        self.signal_decay_rate = getattr(settings, 'signal_decay_rate', 0.7)
        self.activation_threshold = getattr(settings, 'activation_threshold', 0.1)
        self.max_propagation_steps = getattr(settings, 'max_propagation_steps', 5)
        self.concept_extraction_enabled = getattr(settings, 'enable_concept_extraction', True)

        # Pattern completion parameters
        self.pattern_coherence_threshold = 0.6
        self.min_cluster_size = 3
        self.max_answer_nodes = 10

        logger.info("Signal Processor initialized with decay_rate=%.2f, threshold=%.2f, max_steps=%d",
                   self.signal_decay_rate, self.activation_threshold, self.max_propagation_steps)

    def query_to_activation(self, query: str, context: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Convert natural language query to node activation pattern

        Args:
            query: Natural language query
            context: Optional context information

        Returns:
            Dictionary mapping node IDs to activation levels
        """
        try:
            logger.debug(f"Converting query to activation: '{query[:50]}...'")

            # Extract key concepts from query
            concepts = self._extract_concepts(query, context)

            if not concepts:
                logger.warning("No concepts extracted from query")
                return {}

            # Find nodes matching these concepts
            activation = {}
            concept_matches = []

            for concept in concepts:
                matches = self._find_matching_nodes(concept)
                concept_matches.extend(matches)

            # Convert matches to activation levels
            for match in concept_matches:
                node_id = match['node_id']
                similarity = match['similarity']
                activation_level = self._similarity_to_activation(similarity)

                # Combine activations if node matches multiple concepts
                activation[node_id] = max(activation.get(node_id, 0), activation_level)

            logger.debug(f"Query activation complete: {len(activation)} nodes activated")
            return activation

        except Exception as e:
            logger.error(f"Query to activation failed: {e}")
            return {}

    def _extract_concepts(self, query: str, context: Dict[str, Any] = None) -> List[str]:
        """
        Extract key concepts from natural language query

        Args:
            query: Input query
            context: Optional context

        Returns:
            List of extracted concepts
        """
        try:
            if self.llm_interface and self.concept_extraction_enabled:
                # Use LLM for advanced concept extraction
                return self._extract_concepts_with_llm(query, context)
            else:
                # Fallback to simple keyword extraction
                return self._extract_concepts_simple(query)

        except Exception as e:
            logger.warning(f"Concept extraction failed, using fallback: {e}")
            return self._extract_concepts_simple(query)

    def _extract_concepts_with_llm(self, query: str, context: Dict[str, Any] = None) -> List[str]:
        """Use LLM to extract concepts from query"""
        try:
            prompt = f"""Extract the key concepts, entities, and topics from this query.
Return only a comma-separated list of concepts, no other text.

Query: {query}

Concepts:"""

            if self.llm_interface and hasattr(self.llm_interface, 'generate_text'):
                response = self.llm_interface.generate_text(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.1
                ).strip()

                # Parse comma-separated concepts
                concepts = [c.strip() for c in response.split(',') if c.strip()]
                concepts = [c for c in concepts if len(c) > 1]  # Filter out single characters

                logger.debug(f"LLM extracted concepts: {concepts}")
                return concepts[:10]  # Limit to top 10

        except Exception as e:
            logger.warning(f"LLM concept extraction failed: {e}")

        # Fallback
        return self._extract_concepts_simple(query)

    def _extract_concepts_simple(self, query: str) -> List[str]:
        """Simple keyword-based concept extraction"""
        # Basic tokenization and filtering
        words = query.lower().split()
        concepts = []

        # Remove stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who'}

        for word in words:
            word = word.strip('.,!?')
            if len(word) > 2 and word not in stop_words:
                concepts.append(word)

        # Add bigrams for better concept capture
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram) > 5:  # Meaningful bigrams
                concepts.append(bigram)

        logger.debug(f"Simple concept extraction: {concepts[:5]}...")
        return list(set(concepts))[:15]  # Remove duplicates, limit to 15

    def _find_matching_nodes(self, concept: str) -> List[Dict[str, Any]]:
        """
        Find nodes that match a given concept

        Args:
            concept: Concept to match

        Returns:
            List of matching nodes with similarity scores
        """
        matches = []

        try:
            # Get concept embedding
            concept_embedding = self.memory_manager.embedder.encode_texts([concept])[0]

            # Search for similar nodes using vector search
            search_results = self.memory_manager.vector_store.search(concept_embedding, top_k=20)

            for result_tuple in search_results:
                result_id, score, metadata = result_tuple

                # Convert similarity score to our format
                similarity = float(score)

                if similarity > 0.3:  # Minimum similarity threshold
                    matches.append({
                        'node_id': result_id,
                        'similarity': similarity,
                        'metadata': metadata
                    })

        except Exception as e:
            logger.warning(f"Node matching failed for concept '{concept}': {e}")

        return matches[:10]  # Limit matches per concept

    def _similarity_to_activation(self, similarity: float) -> float:
        """Convert similarity score to activation level"""
        # Sigmoid-like transformation to get activation between 0 and 1
        activation = 1 / (1 + np.exp(-5 * (similarity - 0.5)))
        return min(1.0, max(0.0, activation))

    def propagate_signal(self, initial_activation: Dict[str, float],
                        max_steps: int = None, decay: float = None) -> Dict[str, float]:
        """
        Propagate activation through the neural network

        Args:
            initial_activation: Initial node activations
            max_steps: Maximum propagation steps (uses config default if None)
            decay: Signal decay rate (uses config default if None)

        Returns:
            Final activation pattern after propagation
        """
        try:
            max_steps = max_steps or self.max_propagation_steps
            decay = decay or self.signal_decay_rate

            logger.debug(f"Propagating signal with {len(initial_activation)} initial activations, max_steps={max_steps}, decay={decay}")

            current_activation = initial_activation.copy()

            for step in range(max_steps):
                if not current_activation:
                    break

                new_activation = {}
                propagation_count = 0

                # Propagate to neighboring nodes
                for node_id, activation in current_activation.items():
                    if activation < self.activation_threshold:
                        continue

                    # Get neighbors and their connection weights
                    neighbors = self._get_neighbor_weights(node_id)

                    for neighbor_id, edge_weight in neighbors:
                        # Calculate propagated activation
                        propagated_activation = activation * decay * edge_weight

                        if propagated_activation >= self.activation_threshold:
                            # Accumulate activation from multiple sources
                            new_activation[neighbor_id] = max(
                                new_activation.get(neighbor_id, 0),
                                propagated_activation
                            )
                            propagation_count += 1

                current_activation = new_activation

                logger.debug(f"Propagation step {step + 1}: {propagation_count} signals propagated")

                if propagation_count == 0:
                    break  # No more propagation possible

            # Filter final activations above threshold
            final_activation = {
                node_id: activation
                for node_id, activation in current_activation.items()
                if activation >= self.activation_threshold
            }

            logger.debug(f"Signal propagation complete: {len(final_activation)} nodes activated")
            return final_activation

        except Exception as e:
            logger.error(f"Signal propagation failed: {e}")
            return initial_activation

    def _get_neighbor_weights(self, node_id: str) -> List[Tuple[str, float]]:
        """
        Get neighboring nodes and their connection weights

        Args:
            node_id: Source node ID

        Returns:
            List of (neighbor_id, weight) tuples
        """
        neighbors = []

        try:
            # Get adjacency list for this node
            if node_id in self.neural_mesh.adjacency_list:
                for neighbor_id in self.neural_mesh.adjacency_list[node_id]:
                    # Find edge weight
                    edge_key = (node_id, neighbor_id)
                    reverse_key = (neighbor_id, node_id)

                    edge = self.neural_mesh.edges.get(edge_key) or self.neural_mesh.edges.get(reverse_key)

                    if edge:
                        weight = edge.weight
                        # Boost semantic relationships
                        if edge.relationship_label:
                            weight *= 1.2  # 20% boost for semantic edges

                        neighbors.append((neighbor_id, weight))

        except Exception as e:
            logger.warning(f"Failed to get neighbors for {node_id}: {e}")

        return neighbors

    def find_resonant_pattern(self, final_activation: Dict[str, float],
                            query: str = None) -> Dict[str, Any]:
        """
        Extract coherent answer pattern from activation distribution

        Args:
            final_activation: Final activation pattern after propagation
            query: Original query for context

        Returns:
            Pattern completion results
        """
        try:
            logger.debug(f"Finding resonant pattern from {len(final_activation)} activations")

            if not final_activation:
                return {'pattern_found': False, 'reason': 'no_activations'}

            # Sort nodes by activation level
            sorted_nodes = sorted(
                final_activation.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # Extract top activated nodes
            top_nodes = sorted_nodes[:self.max_answer_nodes]

            # Cluster nodes by semantic relationships
            clusters = self._cluster_nodes_by_relationships(top_nodes)

            if not clusters:
                # Fallback: use simple activation ranking
                best_nodes = [node_id for node_id, _ in top_nodes[:5]]
                return {
                    'pattern_found': True,
                    'method': 'activation_ranking',
                    'answer_nodes': best_nodes,
                    'confidence': 0.5,
                    'reason': 'no_relationship_clusters'
                }

            # Find best cluster
            best_cluster = self._select_best_cluster(clusters, query)

            if best_cluster:
                coherence_score = self._calculate_cluster_coherence(best_cluster)

                return {
                    'pattern_found': True,
                    'method': 'relationship_clustering',
                    'answer_nodes': best_cluster['nodes'],
                    'confidence': coherence_score,
                    'cluster_info': best_cluster,
                    'total_clusters': len(clusters)
                }
            else:
                return {
                    'pattern_found': False,
                    'reason': 'no_coherent_clusters',
                    'total_clusters': len(clusters)
                }

        except Exception as e:
            logger.error(f"Pattern completion failed: {e}")
            return {'pattern_found': False, 'reason': f'error: {str(e)}'}

    def _cluster_nodes_by_relationships(self, top_nodes: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
        """
        Cluster nodes based on their semantic relationships

        Args:
            top_nodes: List of (node_id, activation) tuples

        Returns:
            List of clusters with relationship information
        """
        clusters = []
        processed_nodes = set()

        for node_id, activation in top_nodes:
            if node_id in processed_nodes:
                continue

            # Start new cluster with this node
            cluster = {
                'nodes': [node_id],
                'activations': [activation],
                'relationships': [],
                'relationship_types': set()
            }

            # Find connected nodes through relationships
            related_nodes = self._find_related_nodes_through_relationships(node_id, top_nodes)

            for related_id, rel_info in related_nodes:
                if related_id not in processed_nodes:
                    # Find activation for this related node
                    related_activation = next((act for nid, act in top_nodes if nid == related_id), 0)

                    cluster['nodes'].append(related_id)
                    cluster['activations'].append(related_activation)
                    cluster['relationships'].append(rel_info)
                    cluster['relationship_types'].add(rel_info.get('relationship_type', 'unknown'))

                    processed_nodes.add(related_id)

            processed_nodes.add(node_id)

            # Only keep meaningful clusters
            if len(cluster['nodes']) >= self.min_cluster_size:
                cluster['avg_activation'] = sum(cluster['activations']) / len(cluster['activations'])
                cluster['relationship_count'] = len(cluster['relationships'])
                clusters.append(cluster)

        logger.debug(f"Created {len(clusters)} relationship clusters")
        return clusters

    def _find_related_nodes_through_relationships(self, node_id: str,
                                                candidate_nodes: List[Tuple[str, float]]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Find nodes related to the given node through semantic relationships

        Args:
            node_id: Source node
            candidate_nodes: Candidate nodes to consider

        Returns:
            List of (node_id, relationship_info) tuples
        """
        related = []
        candidate_ids = {nid for nid, _ in candidate_nodes}

        # Get semantic relationships for this node
        relationships = self.neural_mesh.get_semantic_relationships(node_id)

        for rel in relationships:
            target_id = rel['target_node'] if rel['direction'] == 'outgoing' else rel['source_node']

            if target_id in candidate_ids:
                related.append((target_id, rel))

        return related

    def _select_best_cluster(self, clusters: List[Dict[str, Any]], query: str = None) -> Optional[Dict[str, Any]]:
        """
        Select the best cluster based on coherence and relevance

        Args:
            clusters: List of clusters to evaluate
            query: Original query for context

        Returns:
            Best cluster or None
        """
        if not clusters:
            return None

        scored_clusters = []

        for cluster in clusters:
            # Calculate coherence score
            coherence = self._calculate_cluster_coherence(cluster)

            # Calculate query relevance if query provided
            relevance = 1.0
            if query:
                relevance = self._calculate_query_relevance(cluster, query)

            # Combined score
            final_score = coherence * relevance

            scored_clusters.append((cluster, final_score))

        # Sort by score and return best
        scored_clusters.sort(key=lambda x: x[1], reverse=True)

        best_cluster, best_score = scored_clusters[0]

        if best_score >= self.pattern_coherence_threshold:
            logger.debug(f"Selected best cluster with score {best_score:.2f}")
            return best_cluster

        return None

    def _calculate_cluster_coherence(self, cluster: Dict[str, Any]) -> float:
        """Calculate coherence score for a cluster"""
        try:
            nodes = cluster['nodes']
            relationships = cluster['relationships']

            if len(nodes) < 2:
                return 0.0

            # Coherence based on relationship density
            max_possible_relationships = len(nodes) * (len(nodes) - 1) / 2
            actual_relationships = len(relationships)

            density_score = actual_relationships / max_possible_relationships if max_possible_relationships > 0 else 0

            # Coherence based on relationship types (prefer diverse but related types)
            type_count = len(cluster['relationship_types'])
            type_diversity = min(type_count / 3, 1.0)  # Cap at 3 types

            # Activation consistency
            activations = cluster['activations']
            activation_std = np.std(activations) if len(activations) > 1 else 0
            activation_consistency = 1.0 / (1.0 + activation_std)  # Lower std = higher consistency

            coherence = (density_score * 0.5 + type_diversity * 0.3 + activation_consistency * 0.2)

            return min(1.0, coherence)

        except Exception as e:
            logger.warning(f"Coherence calculation failed: {e}")
            return 0.0

    def _calculate_query_relevance(self, cluster: Dict[str, Any], query: str) -> float:
        """Calculate how relevant this cluster is to the query"""
        try:
            # Simple relevance based on semantic relationship types
            query_lower = query.lower()

            relevance_score = 0.5  # Base relevance

            # Boost for certain relationship types that match query patterns
            relationship_types = cluster.get('relationship_types', set())

            if 'definitional' in relationship_types and ('what' in query_lower or 'define' in query_lower):
                relevance_score += 0.2

            if 'geographic' in relationship_types and ('where' in query_lower or 'location' in query_lower):
                relevance_score += 0.2

            if 'temporal' in relationship_types and ('when' in query_lower or 'time' in query_lower):
                relevance_score += 0.2

            if 'causal' in relationship_types and ('why' in query_lower or 'because' in query_lower):
                relevance_score += 0.2

            return min(1.0, relevance_score)

        except Exception as e:
            logger.warning(f"Query relevance calculation failed: {e}")
            return 0.5

    def process_query_substrate(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete substrate processing pipeline for a query

        Args:
            query: Natural language query
            context: Optional context

        Returns:
            Substrate processing results
        """
        try:
            start_time = time.time()

            # Step 1: Convert query to activation pattern
            initial_activation = self.query_to_activation(query, context)

            if not initial_activation:
                return {
                    'success': False,
                    'reason': 'no_initial_activation',
                    'processing_time': time.time() - start_time
                }

            # Step 2: Propagate signal through network
            final_activation = self.propagate_signal(initial_activation)

            # Step 3: Find resonant pattern
            pattern_result = self.find_resonant_pattern(final_activation, query)

            processing_time = time.time() - start_time

            result = {
                'success': pattern_result.get('pattern_found', False),
                'method': 'substrate_signal_propagation',
                'initial_activations': len(initial_activation),
                'final_activations': len(final_activation),
                'pattern_result': pattern_result,
                'processing_time': processing_time
            }

            logger.info(f"Substrate processing complete in {processing_time:.2f}s: {result['success']}")
            return result

        except Exception as e:
            logger.error(f"Substrate processing failed: {e}")
            return {
                'success': False,
                'reason': f'error: {str(e)}',
                'processing_time': time.time() - start_time
            }

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get signal processing statistics"""
        return {
            'signal_decay_rate': self.signal_decay_rate,
            'activation_threshold': self.activation_threshold,
            'max_propagation_steps': self.max_propagation_steps,
            'pattern_coherence_threshold': self.pattern_coherence_threshold,
            'concept_extraction_enabled': self.concept_extraction_enabled
        }
