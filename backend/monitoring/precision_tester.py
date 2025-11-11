"""
Precision@K Testing for retrieval evaluation
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class PrecisionTester:
    """
    Tests retrieval precision@K for memory system evaluation
    """

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    def test_precision_at_k(self, queries: List[str], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Test precision@K for a set of queries

        Args:
            queries: List of test queries
            k_values: K values to test (e.g., [1, 3, 5, 10])

        Returns:
            Precision results for each K
        """
        results = {}

        for k in k_values:
            precisions = []

            for query in queries:
                precision = self._calculate_precision_at_k(query, k)
                precisions.append(precision)

            avg_precision = np.mean(precisions) if precisions else 0
            std_precision = np.std(precisions) if precisions else 0

            results[f'precision@{k}'] = {
                'average': avg_precision,
                'std_dev': std_precision,
                'min': min(precisions) if precisions else 0,
                'max': max(precisions) if precisions else 0,
                'query_count': len(queries)
            }

        return results

    def _calculate_precision_at_k(self, query: str, k: int) -> float:
        """
        Calculate precision@K for a single query

        Args:
            query: Test query
            k: Number of results to consider

        Returns:
            Precision@K score (0-1)
        """
        try:
            # Get retrieval results
            results = self.memory_manager.hybrid_search(query, max_results=k)

            if not results or 'results' not in results:
                return 0.0

            retrieved_chunks = results['results'][:k]

            # For evaluation, we need ground truth relevance
            # This is a simplified version - in practice, you'd have human-labeled relevant chunks
            relevant_count = self._estimate_relevance(query, retrieved_chunks)

            precision = relevant_count / k if k > 0 else 0

            return precision

        except Exception as e:
            logger.error(f"Failed to calculate precision@K for query '{query}': {e}")
            return 0.0

    def _estimate_relevance(self, query: str, chunks: List[Dict[str, Any]]) -> int:
        """
        Estimate how many chunks are relevant to the query
        This is a heuristic - real evaluation would use human labels
        """
        relevant_count = 0

        query_terms = set(query.lower().split())

        for chunk in chunks:
            chunk_text = chunk.get('text', '').lower()
            chunk_terms = set(chunk_text.split())

            # Calculate Jaccard similarity
            intersection = len(query_terms & chunk_terms)
            union = len(query_terms | chunk_terms)

            if union > 0:
                jaccard = intersection / union
                if jaccard > 0.1:  # Relevance threshold
                    relevant_count += 1

        return relevant_count

    def generate_test_queries(self, sample_size: int = 50) -> List[str]:
        """
        Generate test queries from existing memory

        Args:
            sample_size: Number of queries to generate

        Returns:
            List of test queries
        """
        try:
            # Get sample memory chunks
            chunks = self.memory_manager.list_memory_chunks(limit=sample_size * 2)

            queries = []

            for chunk in chunks[:sample_size]:
                text = chunk.get('text', '')

                # Generate query from chunk content
                # Simple approach: extract first sentence or key phrases
                sentences = text.split('.')[:2]  # First 1-2 sentences
                query = '.'.join(sentences).strip()

                if len(query) > 20:  # Minimum length
                    queries.append(query)

            return queries

        except Exception as e:
            logger.error(f"Failed to generate test queries: {e}")
            return []
