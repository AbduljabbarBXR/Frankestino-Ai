"""
Predictive Association Engine
Provides semantic prediction, completion, and search capabilities
"""
import logging
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import heapq
import numpy as np

from .autonomous_mesh import AutonomousMesh
from .connection_manager import ConnectionManager
from ..utils.cache import get_cache

logger = logging.getLogger(__name__)


class PredictionConfig:
    """Configuration for prediction behavior"""

    def __init__(self):
        self.max_predictions = 20
        self.min_confidence = 0.1
        self.context_window_size = 5
        self.temporal_decay_factor = 0.95  # daily decay
        self.indirect_association_weight = 0.3
        self.co_occurrence_boost = 0.2
        self.semantic_search_threshold = 0.6
        self.analogy_depth = 3  # max analogy chain length


class AssociationEngine:
    """
    Engine for predicting word associations, semantic completion,
    and intelligent search within the autonomous word network
    """

    def __init__(self, autonomous_mesh: AutonomousMesh, connection_manager: ConnectionManager):
        self.mesh = autonomous_mesh
        self.connection_manager = connection_manager
        self.config = PredictionConfig()

        # Prediction caching
        self.cache = get_cache()
        self.prediction_cache = {}  # (word, context_hash) -> predictions

        # Statistics
        self.prediction_stats = {
            "total_predictions": 0,
            "cache_hits": 0,
            "semantic_searches": 0,
            "analogies_generated": 0,
            "average_prediction_time": 0.0
        }

        logger.info("Association Engine initialized")

    def predict_associations(self, word: str, context: List[str] = None,
                           top_k: int = None, min_confidence: float = None) -> List[Tuple[str, float, str]]:
        """
        Predict associated words for a given word

        Args:
            word: Target word to predict associations for
            context: List of context words
            top_k: Number of predictions to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of (word, confidence, association_type) tuples
        """
        if top_k is None:
            top_k = self.config.max_predictions
        if min_confidence is None:
            min_confidence = self.config.min_confidence

        start_time = time.time()

        # Check cache first
        context_hash = self._hash_context(context or [])
        cache_key = f"predict_{word}_{context_hash}"

        cached = self.cache.get(cache_key)
        if cached:
            self.prediction_stats["cache_hits"] += 1
            return cached[:top_k]

        # Get direct associations
        direct_associations = self._get_direct_associations(word, context)

        # Get indirect associations through network traversal
        indirect_associations = self._get_indirect_associations(word, context)

        # Combine and rank associations
        all_associations = self._combine_associations(direct_associations, indirect_associations)

        # Filter by confidence and sort
        filtered_associations = [
            assoc for assoc in all_associations
            if assoc[1] >= min_confidence
        ]

        filtered_associations.sort(key=lambda x: x[1], reverse=True)
        result = filtered_associations[:top_k]

        # Cache the result
        self.cache.set(cache_key, result, ttl=300)  # 5 minutes

        # Update statistics
        prediction_time = time.time() - start_time
        self._update_prediction_stats(prediction_time)

        self.prediction_stats["total_predictions"] += 1

        return result

    def _hash_context(self, context: List[str]) -> str:
        """Create a hash for context caching"""
        return hash(tuple(sorted(context))) if context else "no_context"

    def _get_direct_associations(self, word: str, context: List[str] = None) -> List[Tuple[str, float, str]]:
        """Get direct associations from the network"""
        if word not in self.mesh.word_neurons:
            return []

        neuron_id = self.mesh.word_neurons[word]
        associations = []

        # Get all direct connections
        for neighbor_id in self.mesh.adjacency_list.get(neuron_id, []):
            edge = self.mesh._get_edge(neuron_id, neighbor_id)
            if edge and edge.weight > 0:
                neighbor_word = self.mesh.neuron_words.get(neighbor_id)
                if neighbor_word and neighbor_word != word:
                    # Apply context boosting if context provided
                    confidence = edge.weight
                    if context:
                        confidence = self._apply_context_boost(confidence, neighbor_word, context)

                    associations.append((neighbor_word, confidence, "direct"))

        return associations

    def _get_indirect_associations(self, word: str, context: List[str] = None,
                                 max_depth: int = 2) -> List[Tuple[str, float, str]]:
        """Get indirect associations through network traversal"""
        if word not in self.mesh.word_neurons:
            return []

        neuron_id = self.mesh.word_neurons[word]
        associations = defaultdict(float)

        # Traverse network up to max_depth
        visited = set()
        queue = [(neuron_id, 0, 1.0)]  # (neuron_id, depth, path_weight)

        while queue:
            current_id, depth, path_weight = queue.pop(0)

            if current_id in visited or depth >= max_depth:
                continue

            visited.add(current_id)

            # Process neighbors
            for neighbor_id in self.mesh.adjacency_list.get(current_id, []):
                if neighbor_id in visited:
                    continue

                edge = self.mesh._get_edge(current_id, neighbor_id)
                if not edge or edge.weight <= 0:
                    continue

                neighbor_word = self.mesh.neuron_words.get(neighbor_id)
                if neighbor_word and neighbor_word != word:
                    # Calculate indirect association strength
                    indirect_weight = path_weight * edge.weight * self.config.indirect_association_weight

                    # Apply distance decay
                    indirect_weight *= (0.8 ** depth)

                    # Apply context boosting
                    if context:
                        indirect_weight = self._apply_context_boost(indirect_weight, neighbor_word, context)

                    associations[neighbor_word] = max(associations[neighbor_word], indirect_weight)

                    # Continue traversal if not too deep
                    if depth < max_depth - 1:
                        queue.append((neighbor_id, depth + 1, path_weight * edge.weight))

        # Convert to list format
        return [(word, weight, "indirect") for word, weight in associations.items()]

    def _apply_context_boost(self, base_weight: float, target_word: str, context: List[str]) -> float:
        """Apply context-based boosting to association weight"""
        if not context:
            return base_weight

        boost_factor = 1.0

        # Check if target word appears in context
        if target_word in context:
            boost_factor += self.config.co_occurrence_boost

        # Check semantic similarity with context words
        context_similarities = []
        for context_word in context:
            if context_word in self.mesh.word_neurons:
                context_neuron = self.mesh.word_neurons[context_word]
                target_neuron = self.mesh.word_neurons.get(target_word)

                if target_neuron:
                    edge = self.mesh._get_edge(context_neuron, target_neuron)
                    if edge:
                        context_similarities.append(edge.weight)

        if context_similarities:
            avg_context_similarity = sum(context_similarities) / len(context_similarities)
            boost_factor += avg_context_similarity * 0.3

        return base_weight * boost_factor

    def _combine_associations(self, direct: List[Tuple[str, float, str]],
                            indirect: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        """Combine direct and indirect associations"""
        combined = defaultdict(lambda: [0.0, ""])

        # Process direct associations
        for word, weight, assoc_type in direct:
            if weight > combined[word][0]:
                combined[word] = [weight, assoc_type]

        # Process indirect associations (with lower priority)
        for word, weight, assoc_type in indirect:
            if word not in combined or weight > combined[word][0] * 0.7:  # Allow indirect if significantly better
                combined[word] = [weight, assoc_type]

        return [(word, weight, assoc_type) for word, (weight, assoc_type) in combined.items()]

    def semantic_search(self, query: str, top_k: int = 10,
                       search_type: str = "similar") -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform semantic search for words similar to query

        Args:
            query: Search query (word or phrase)
            top_k: Number of results to return
            search_type: Type of search ("similar", "related", "broader", "narrower")

        Returns:
            List of (word, similarity, metadata) tuples
        """
        start_time = time.time()

        # Tokenize query
        query_words = self._tokenize_query(query)
        if not query_words:
            return []

        results = []

        if len(query_words) == 1:
            # Single word search
            results = self._single_word_search(query_words[0], search_type, top_k)
        else:
            # Multi-word search
            results = self._multi_word_search(query_words, search_type, top_k)

        # Update statistics
        search_time = time.time() - start_time
        self._update_prediction_stats(search_time)
        self.prediction_stats["semantic_searches"] += 1

        return results

    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize search query"""
        words = query.lower().split()
        # Filter out stop words and short words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [word for word in words if len(word) > 1 and word not in stop_words]

    def _single_word_search(self, word: str, search_type: str, top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for words related to a single query word"""
        if word not in self.mesh.word_neurons:
            return []

        query_neuron = self.mesh.word_neurons[word]
        candidates = []

        # Search through all neurons
        for candidate_word, candidate_neuron_id in self.mesh.word_neurons.items():
            if candidate_word == word:
                continue

            candidate_neuron = self.mesh.nodes.get(candidate_neuron_id)
            if not candidate_neuron or not hasattr(candidate_neuron, 'embedding') or not candidate_neuron.embedding:
                continue

            # Calculate relevance based on search type
            relevance, metadata = self._calculate_search_relevance(
                query_neuron, candidate_neuron_id, search_type
            )

            if relevance >= self.config.semantic_search_threshold:
                candidates.append((candidate_word, relevance, metadata))

        # Sort and return top results
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def _calculate_search_relevance(self, query_neuron_id: str, candidate_neuron_id: str,
                                  search_type: str) -> Tuple[float, Dict[str, Any]]:
        """Calculate relevance score for search candidate"""
        metadata = {"search_type": search_type}

        # Get direct connection if exists
        edge = self.mesh._get_edge(query_neuron_id, candidate_neuron_id)
        if edge:
            metadata["direct_connection"] = True
            metadata["connection_type"] = edge.connection_type
            metadata["connection_weight"] = edge.weight

            if search_type == "similar":
                return edge.weight, metadata
            elif search_type == "related":
                return edge.weight * 0.9, metadata

        # Calculate embedding similarity
        query_neuron = self.mesh.nodes.get(query_neuron_id)
        candidate_neuron = self.mesh.nodes.get(candidate_neuron_id)

        if (query_neuron and candidate_neuron and
            hasattr(query_neuron, 'embedding') and query_neuron.embedding and
            hasattr(candidate_neuron, 'embedding') and candidate_neuron.embedding):

            similarity = self._cosine_similarity(query_neuron.embedding, candidate_neuron.embedding)
            metadata["embedding_similarity"] = similarity

            # Adjust based on search type
            if search_type == "similar":
                return similarity, metadata
            elif search_type == "related":
                return similarity * 0.8, metadata

        return 0.0, metadata

    def _multi_word_search(self, query_words: List[str], search_type: str, top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for words related to multiple query words"""
        if not query_words:
            return []

        # Get associations for each query word
        all_candidates = defaultdict(lambda: {"total_score": 0.0, "sources": [], "metadata": {}})

        for query_word in query_words:
            associations = self.predict_associations(query_word, query_words, top_k=50)

            for assoc_word, confidence, assoc_type in associations:
                all_candidates[assoc_word]["total_score"] += confidence
                all_candidates[assoc_word]["sources"].append({
                    "query_word": query_word,
                    "confidence": confidence,
                    "type": assoc_type
                })

        # Convert to final format
        results = []
        for word, data in all_candidates.items():
            avg_score = data["total_score"] / len(query_words)
            metadata = {
                "multi_word_search": True,
                "query_words": query_words,
                "sources": data["sources"],
                "average_score": avg_score
            }
            results.append((word, avg_score, metadata))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def generate_analogy(self, word_a: str, word_b: str, word_c: str,
                        top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Generate analogies of the form: word_a : word_b :: word_c : ?

        Args:
            word_a, word_b, word_c: Words in the analogy
            top_k: Number of results to return

        Returns:
            List of (word_d, confidence, explanation) tuples
        """
        start_time = time.time()

        # Get relationships between A-B and find similar pattern for C-?
        analogies = []

        # Get associations for word_b (what B is related to)
        b_associations = self.predict_associations(word_b, top_k=50)

        # Get associations for word_c (potential candidates for D)
        c_associations = dict(self.predict_associations(word_c, top_k=100))

        # Find candidates that maintain the relationship pattern
        for candidate_d, c_confidence in c_associations.items():
            if candidate_d in [word_a, word_b, word_c]:
                continue

            # Check if this candidate has similar relationship strength to C
            # as B has to A
            analogy_score = self._calculate_analogy_score(word_a, word_b, word_c, candidate_d)

            if analogy_score > 0.1:
                explanation = {
                    "analogy": f"{word_a}:{word_b}::{word_c}:{candidate_d}",
                    "relationship_strength": analogy_score,
                    "c_confidence": c_confidence
                }
                analogies.append((candidate_d, analogy_score, explanation))

        analogies.sort(key=lambda x: x[1], reverse=True)

        # Update statistics
        analogy_time = time.time() - start_time
        self._update_prediction_stats(analogy_time)
        self.prediction_stats["analogies_generated"] += 1

        return analogies[:top_k]

    def _calculate_analogy_score(self, a: str, b: str, c: str, d: str) -> float:
        """Calculate how well D completes the analogy A:B::C:D"""
        score = 0.0

        # Method 1: Direct relationship comparison
        if (a in self.mesh.word_neurons and b in self.mesh.word_neurons and
            c in self.mesh.word_neurons and d in self.mesh.word_neurons):

            # Get relationship strengths
            ab_edge = self.mesh._get_edge(self.mesh.word_neurons[a], self.mesh.word_neurons[b])
            cd_edge = self.mesh._get_edge(self.mesh.word_neurons[c], self.mesh.word_neurons[d])

            if ab_edge and cd_edge:
                # How similar are the A-B and C-D relationships?
                relationship_similarity = 1.0 - abs(ab_edge.weight - cd_edge.weight)
                score += relationship_similarity * 0.6

        # Method 2: Semantic vector analogy
        a_neuron = self.mesh.nodes.get(self.mesh.word_neurons.get(a))
        b_neuron = self.mesh.nodes.get(self.mesh.word_neurons.get(b))
        c_neuron = self.mesh.nodes.get(self.mesh.word_neurons.get(c))
        d_neuron = self.mesh.nodes.get(self.mesh.word_neurons.get(d))

        if (a_neuron and b_neuron and c_neuron and d_neuron and
            hasattr(a_neuron, 'embedding') and a_neuron.embedding and
            hasattr(b_neuron, 'embedding') and b_neuron.embedding and
            hasattr(c_neuron, 'embedding') and c_neuron.embedding and
            hasattr(d_neuron, 'embedding') and d_neuron.embedding):

            # Vector analogy: B - A + C should be close to D
            try:
                a_vec = np.array(a_neuron.embedding)
                b_vec = np.array(b_neuron.embedding)
                c_vec = np.array(c_neuron.embedding)
                d_vec = np.array(d_neuron.embedding)

                # Calculate analogy vector: B - A + C
                analogy_vector = b_vec - a_vec + c_vec

                # Calculate similarity to D
                vector_similarity = self._cosine_similarity(analogy_vector, d_vec)
                score += vector_similarity * 0.4
            except:
                pass

        return score

    def complete_phrase(self, partial_phrase: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Complete a partial phrase using learned associations

        Args:
            partial_phrase: Partial text to complete
            top_k: Number of completions to return

        Returns:
            List of (completion, confidence, method) tuples
        """
        words = self._tokenize_query(partial_phrase)
        if not words:
            return []

        completions = []

        # Method 1: Next word prediction based on last word
        if words:
            last_word = words[-1]
            next_words = self.predict_associations(last_word, words[:-1], top_k=top_k*2)

            for word, confidence, assoc_type in next_words:
                completion = partial_phrase + " " + word
                completions.append((completion, confidence * 0.8, "next_word"))

        # Method 2: Semantic completion based on overall meaning
        phrase_embedding = self._get_phrase_embedding(words)
        if phrase_embedding:
            semantic_completions = self._find_semantic_completions(phrase_embedding, top_k=top_k*2)

            for completion, confidence in semantic_completions:
                completions.append((completion, confidence * 0.6, "semantic"))

        # Remove duplicates and sort
        seen = set()
        unique_completions = []
        for completion, confidence, method in completions:
            if completion not in seen:
                unique_completions.append((completion, confidence, method))
                seen.add(completion)

        unique_completions.sort(key=lambda x: x[1], reverse=True)
        return unique_completions[:top_k]

    def _get_phrase_embedding(self, words: List[str]) -> Optional[List[float]]:
        """Get average embedding for a phrase"""
        embeddings = []

        for word in words:
            if word in self.mesh.word_neurons:
                neuron = self.mesh.nodes.get(self.mesh.word_neurons[word])
                if neuron and hasattr(neuron, 'embedding') and neuron.embedding:
                    embeddings.append(neuron.embedding)

        if not embeddings:
            return None

        # Return average embedding
        return np.mean(embeddings, axis=0).tolist()

    def _find_semantic_completions(self, phrase_embedding: List[float], top_k: int) -> List[Tuple[str, float]]:
        """Find semantic completions for a phrase embedding"""
        completions = []

        # Search for words whose embeddings are similar to the phrase
        for word, neuron_id in self.mesh.word_neurons.items():
            neuron = self.mesh.nodes.get(neuron_id)
            if neuron and hasattr(neuron, 'embedding') and neuron.embedding:
                similarity = self._cosine_similarity(phrase_embedding, neuron.embedding)
                if similarity > 0.7:  # High similarity threshold
                    completions.append((word, similarity))

        completions.sort(key=lambda x: x[1], reverse=True)
        return completions[:top_k]

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

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction engine statistics"""
        return {
            **self.prediction_stats,
            "cache_size": len(self.prediction_cache),
            "config": {
                "max_predictions": self.config.max_predictions,
                "min_confidence": self.config.min_confidence,
                "semantic_search_threshold": self.config.semantic_search_threshold
            }
        }

    def _update_prediction_stats(self, operation_time: float):
        """Update prediction timing statistics"""
        if self.prediction_stats["total_predictions"] == 0:
            self.prediction_stats["average_prediction_time"] = operation_time
        else:
            # Rolling average
            alpha = 0.1
            self.prediction_stats["average_prediction_time"] = (
                alpha * operation_time +
                (1 - alpha) * self.prediction_stats["average_prediction_time"]
            )

    def __del__(self):
        """Cleanup"""
        pass
