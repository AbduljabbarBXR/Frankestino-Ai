"""
Pattern Recognition Engine
Learns successful query patterns and optimizes memory retrieval
Part of Scaffolding & Substrate Model - Phase 1
"""
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import json
from pathlib import Path
import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


class PatternRecognitionEngine:
    """
    Learns and recognizes patterns in successful memory retrieval and query processing
    Implements background learning for continuous memory optimization
    """

    def __init__(self, neural_mesh, memory_manager):
        """
        Initialize pattern recognition engine

        Args:
            neural_mesh: NeuralMesh instance for pattern storage
            memory_manager: MemoryManager instance for data access
        """
        self.neural_mesh = neural_mesh
        self.memory_manager = memory_manager

        # Pattern storage
        self.query_patterns = {}  # pattern_id -> pattern_data
        self.success_patterns = defaultdict(list)  # query_type -> successful_patterns
        self.retrieval_patterns = defaultdict(list)  # category -> retrieval_patterns

        # Performance tracking
        self.pattern_performance = {}  # pattern_id -> performance_metrics
        self.query_history = []  # Recent query history for analysis

        # Learning parameters
        self.min_pattern_confidence = 0.6
        self.max_patterns_per_type = 50
        self.pattern_decay_rate = 0.95  # Daily decay for pattern relevance

        # Storage path
        self.storage_path = settings.data_dir / "memory" / "pattern_recognition.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing patterns
        self._load_patterns()

        logger.info("Pattern Recognition Engine initialized")

    def _load_patterns(self):
        """Load patterns from disk"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.query_patterns = data.get('query_patterns', {})
                self.success_patterns = defaultdict(list, data.get('success_patterns', {}))
                self.retrieval_patterns = defaultdict(list, data.get('retrieval_patterns', {}))
                self.pattern_performance = data.get('pattern_performance', {})

                logger.info(f"Loaded {len(self.query_patterns)} patterns from disk")

            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")
        else:
            logger.info("No existing patterns found, starting fresh")

    def _save_patterns(self):
        """Save patterns to disk"""
        try:
            data = {
                'query_patterns': self.query_patterns,
                'success_patterns': dict(self.success_patterns),
                'retrieval_patterns': dict(self.retrieval_patterns),
                'pattern_performance': self.pattern_performance,
                'metadata': {
                    'created_at': time.time(),
                    'total_patterns': len(self.query_patterns)
                }
            }

            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug("Patterns saved to disk")

        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")

    def analyze_query_success(self, query: str, memory_results: Dict[str, Any],
                            response_quality: float, execution_time: float) -> Dict[str, Any]:
        """
        Analyze a completed query and extract successful patterns

        Args:
            query: The user query
            memory_results: Memory search results
            response_quality: Quality score of the response (0.0-1.0)
            execution_time: Time taken to process query

        Returns:
            Analysis results with extracted patterns
        """
        try:
            # Extract query features
            query_features = self._extract_query_features(query)

            # Analyze memory retrieval effectiveness
            retrieval_effectiveness = self._analyze_retrieval_effectiveness(memory_results)

            # Determine if this was a successful pattern
            is_successful = response_quality >= 0.7 and retrieval_effectiveness >= 0.6

            pattern_data = {
                'query': query,
                'query_features': query_features,
                'memory_chunks_used': len(memory_results.get('results', [])),
                'categories_searched': memory_results.get('categories', []),
                'search_type': memory_results.get('search_type', 'unknown'),
                'response_quality': response_quality,
                'retrieval_effectiveness': retrieval_effectiveness,
                'execution_time': execution_time,
                'timestamp': time.time(),
                'is_successful': is_successful
            }

            # Store successful patterns for learning
            if is_successful:
                query_type = query_features.get('query_type', 'general')
                self.success_patterns[query_type].append(pattern_data)

                # Limit patterns per type
                if len(self.success_patterns[query_type]) > self.max_patterns_per_type:
                    # Keep most recent patterns
                    self.success_patterns[query_type] = self.success_patterns[query_type][-self.max_patterns_per_type:]

                # Create pattern ID and store
                pattern_id = f"pattern_{int(time.time())}_{hash(query) % 10000}"
                self.query_patterns[pattern_id] = pattern_data
                self.pattern_performance[pattern_id] = {
                    'success_rate': 1.0,  # Initial success
                    'avg_quality': response_quality,
                    'usage_count': 1,
                    'last_used': time.time()
                }

                logger.debug(f"Learned successful pattern: {pattern_id} (type: {query_type})")

            # Add to recent history for analysis
            self.query_history.append(pattern_data)
            if len(self.query_history) > 100:  # Keep last 100 queries
                self.query_history = self.query_history[-100:]

            # Save patterns periodically
            if len(self.query_patterns) % 10 == 0:  # Save every 10 new patterns
                self._save_patterns()

            return {
                'pattern_extracted': is_successful,
                'query_type': query_features.get('query_type', 'unknown'),
                'success_score': response_quality * retrieval_effectiveness,
                'pattern_data': pattern_data
            }

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {'error': str(e)}

    def _extract_query_features(self, query: str) -> Dict[str, Any]:
        """Extract features from a query for pattern recognition"""
        query_lower = query.lower().strip()

        # Query type classification
        query_type = 'general'

        # Question detection
        if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']):
            query_type = 'question'

        # Command detection
        elif any(word in query_lower for word in ['create', 'add', 'delete', 'update', 'show', 'list', 'find']):
            query_type = 'command'

        # Search detection
        elif any(word in query_lower for word in ['search', 'look for', 'find']):
            query_type = 'search'

        # Conversational detection
        elif len(query.split()) < 5:
            query_type = 'conversational'

        # Extract keywords (simple approach)
        words = query_lower.split()
        keywords = [word for word in words if len(word) > 3 and word not in ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'that', 'this', 'with', 'from', 'about']]

        return {
            'query_type': query_type,
            'length': len(query),
            'word_count': len(words),
            'keywords': keywords[:5],  # Top 5 keywords
            'has_question_mark': '?' in query,
            'is_capitalized': query[0].isupper() if query else False
        }

    def _analyze_retrieval_effectiveness(self, memory_results: Dict[str, Any]) -> float:
        """Analyze how effective the memory retrieval was"""
        results = memory_results.get('results', [])

        if not results:
            return 0.0

        # Calculate average relevance score
        total_score = sum(result.get('score', 0.5) for result in results)
        avg_score = total_score / len(results)

        # Bonus for mesh-enhanced results
        mesh_bonus = 0.1 if memory_results.get('mesh_enhanced', False) else 0.0

        # Penalty for too few results
        result_penalty = 0.0 if len(results) >= 3 else (3 - len(results)) * 0.1

        effectiveness = min(1.0, avg_score + mesh_bonus - result_penalty)

        return max(0.0, effectiveness)

    def get_query_suggestions(self, current_query: str, context: str = None) -> List[Dict[str, Any]]:
        """
        Suggest query improvements based on learned patterns

        Args:
            current_query: Current user query
            context: Optional context about the query

        Returns:
            List of suggested query improvements
        """
        try:
            suggestions = []
            query_features = self._extract_query_features(current_query)
            query_type = query_features.get('query_type', 'general')

            # Find similar successful patterns
            similar_patterns = self._find_similar_patterns(current_query, query_type)

            for pattern in similar_patterns[:3]:  # Top 3 suggestions
                if pattern.get('response_quality', 0) > 0.8:
                    suggestion = {
                        'type': 'similar_successful_query',
                        'description': f"Try rephrasing like: '{pattern['query'][:50]}...'",
                        'expected_improvement': pattern.get('response_quality', 0.8),
                        'pattern_source': pattern.get('query', '')[:30]
                    }
                    suggestions.append(suggestion)

            # Category-specific suggestions
            if context and 'category' in context.lower():
                category_suggestion = {
                    'type': 'category_focus',
                    'description': 'Consider specifying a category for more targeted results',
                    'expected_improvement': 0.7
                }
                suggestions.append(category_suggestion)

            return suggestions

        except Exception as e:
            logger.error(f"Query suggestion failed: {e}")
            return []

    def _find_similar_patterns(self, query: str, query_type: str) -> List[Dict[str, Any]]:
        """Find patterns similar to the current query"""
        similar_patterns = []

        # Get patterns of the same type
        type_patterns = self.success_patterns.get(query_type, [])

        for pattern in type_patterns:
            # Simple similarity based on keyword overlap
            query_keywords = set(self._extract_query_features(query).get('keywords', []))
            pattern_keywords = set(pattern.get('query_features', {}).get('keywords', []))

            if query_keywords and pattern_keywords:
                overlap = len(query_keywords.intersection(pattern_keywords))
                similarity = overlap / len(query_keywords.union(pattern_keywords))

                if similarity > 0.3:  # At least 30% keyword overlap
                    pattern_copy = pattern.copy()
                    pattern_copy['similarity_score'] = similarity
                    similar_patterns.append(pattern_copy)

        # Sort by similarity and quality
        similar_patterns.sort(key=lambda x: x.get('similarity_score', 0) * x.get('response_quality', 0), reverse=True)

        return similar_patterns

    def optimize_memory_retrieval(self, query: str, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest optimizations for memory retrieval based on learned patterns

        Args:
            query: The query being processed
            current_results: Current memory search results

        Returns:
            Optimization suggestions
        """
        try:
            optimizations = {
                'suggested_categories': [],
                'recommended_search_type': 'hybrid',
                'expected_improvement': 0.0
            }

            query_features = self._extract_query_features(query)
            query_type = query_features.get('query_type', 'general')

            # Analyze successful patterns for this query type
            successful_patterns = self.success_patterns.get(query_type, [])

            if successful_patterns:
                # Find patterns with high effectiveness
                effective_patterns = [p for p in successful_patterns if p.get('retrieval_effectiveness', 0) > 0.8]

                if effective_patterns:
                    # Extract common categories from successful patterns
                    categories = []
                    for pattern in effective_patterns:
                        categories.extend(pattern.get('categories_searched', []))

                    if categories:
                        category_counts = Counter(categories)
                        top_categories = [cat for cat, count in category_counts.most_common(3)]
                        optimizations['suggested_categories'] = top_categories

                    # Calculate expected improvement
                    avg_quality = sum(p.get('response_quality', 0.7) for p in effective_patterns) / len(effective_patterns)
                    optimizations['expected_improvement'] = min(0.3, avg_quality - 0.5)  # Cap at 30% improvement

            return optimizations

        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return {'error': str(e)}

    def run_background_learning(self) -> Dict[str, Any]:
        """
        Run background pattern learning and optimization
        Part of Scaffolding & Substrate Model continuous learning
        """
        try:
            logger.info("Running background pattern learning...")

            # Analyze recent query history
            recent_patterns = self.query_history[-20:]  # Last 20 queries

            if not recent_patterns:
                return {'status': 'no_recent_queries'}

            # Identify emerging patterns
            pattern_clusters = self._cluster_similar_queries(recent_patterns)

            # Update pattern performance metrics
            self._update_pattern_performance()

            # Decay old pattern relevance
            self._decay_pattern_relevance()

            # Generate insights
            insights = self._generate_learning_insights(pattern_clusters)

            # Save updated patterns
            self._save_patterns()

            stats = {
                'patterns_analyzed': len(recent_patterns),
                'clusters_found': len(pattern_clusters),
                'insights_generated': len(insights),
                'total_patterns': len(self.query_patterns),
                'background_learning_completed': True
            }

            logger.info(f"Background learning completed: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Background learning failed: {e}")
            return {'error': str(e)}

    def _cluster_similar_queries(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster similar queries for pattern analysis"""
        clusters = []

        for pattern in patterns:
            # Simple clustering based on query type and keywords
            query_type = pattern.get('query_features', {}).get('query_type', 'general')
            keywords = set(pattern.get('query_features', {}).get('keywords', []))

            # Find existing cluster or create new one
            matched_cluster = None
            for cluster in clusters:
                cluster_keywords = set()
                for p in cluster['patterns']:
                    cluster_keywords.update(p.get('query_features', {}).get('keywords', []))

                # Check if keywords overlap significantly
                overlap = len(keywords.intersection(cluster_keywords))
                union = len(keywords.union(cluster_keywords))

                if union > 0 and (overlap / union) > 0.4:  # 40% keyword overlap
                    matched_cluster = cluster
                    break

            if matched_cluster:
                matched_cluster['patterns'].append(pattern)
            else:
                clusters.append({
                    'query_type': query_type,
                    'patterns': [pattern],
                    'avg_quality': pattern.get('response_quality', 0.5)
                })

        # Update cluster averages
        for cluster in clusters:
            qualities = [p.get('response_quality', 0.5) for p in cluster['patterns']]
            cluster['avg_quality'] = sum(qualities) / len(qualities)
            cluster['pattern_count'] = len(cluster['patterns'])

        return clusters

    def _update_pattern_performance(self):
        """Update performance metrics for existing patterns"""
        current_time = time.time()

        for pattern_id, performance in self.pattern_performance.items():
            # Decay usage over time
            time_since_last_use = current_time - performance.get('last_used', current_time)
            decay_factor = 0.99 ** (time_since_last_use / (24 * 60 * 60))  # Daily decay

            performance['usage_count'] *= decay_factor

            # Mark as stale if not used recently
            if time_since_last_use > (30 * 24 * 60 * 60):  # 30 days
                performance['is_stale'] = True

    def _decay_pattern_relevance(self):
        """Decay relevance of old patterns"""
        current_time = time.time()
        cutoff_time = current_time - (90 * 24 * 60 * 60)  # 90 days

        # Remove very old patterns
        old_patterns = []
        for pattern_id, pattern in self.query_patterns.items():
            if pattern.get('timestamp', 0) < cutoff_time:
                old_patterns.append(pattern_id)

        for pattern_id in old_patterns:
            del self.query_patterns[pattern_id]
            self.pattern_performance.pop(pattern_id, None)

        if old_patterns:
            logger.info(f"Decayed {len(old_patterns)} old patterns")

    def _generate_learning_insights(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate insights from pattern clusters"""
        insights = []

        # Find high-performing clusters
        high_perf_clusters = [c for c in clusters if c.get('avg_quality', 0) > 0.8 and c.get('pattern_count', 0) >= 3]

        for cluster in high_perf_clusters:
            insight = {
                'type': 'successful_query_pattern',
                'query_type': cluster.get('query_type', 'unknown'),
                'pattern_count': cluster.get('pattern_count', 0),
                'avg_quality': cluster.get('avg_quality', 0.8),
                'recommendation': f"Prioritize {cluster['query_type']} queries with similar keyword patterns",
                'timestamp': time.time()
            }
            insights.append(insight)

        # Find low-performing clusters
        low_perf_clusters = [c for c in clusters if c.get('avg_quality', 0) < 0.5 and c.get('pattern_count', 0) >= 2]

        for cluster in low_perf_clusters:
            insight = {
                'type': 'problematic_query_pattern',
                'query_type': cluster.get('query_type', 'unknown'),
                'pattern_count': cluster.get('pattern_count', 0),
                'avg_quality': cluster.get('avg_quality', 0.5),
                'recommendation': f"Investigate and improve handling of {cluster['query_type']} queries",
                'timestamp': time.time()
            }
            insights.append(insight)

        return insights

    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get comprehensive pattern recognition statistics"""
        total_patterns = len(self.query_patterns)
        successful_patterns = sum(len(patterns) for patterns in self.success_patterns.values())

        # Calculate pattern type distribution
        type_distribution = {}
        for pattern_type, patterns in self.success_patterns.items():
            type_distribution[pattern_type] = len(patterns)

        # Performance metrics
        if self.pattern_performance:
            avg_success_rate = sum(p.get('success_rate', 0) for p in self.pattern_performance.values()) / len(self.pattern_performance)
            avg_quality = sum(p.get('avg_quality', 0) for p in self.pattern_performance.values()) / len(self.pattern_performance)
        else:
            avg_success_rate = 0.0
            avg_quality = 0.0

        return {
            'total_patterns': total_patterns,
            'successful_patterns': successful_patterns,
            'pattern_types': type_distribution,
            'avg_success_rate': avg_success_rate,
            'avg_quality_score': avg_quality,
            'recent_queries_analyzed': len(self.query_history),
            'background_learning_active': True
        }

    def cleanup(self):
        """Clean up resources and save final state"""
        self._save_patterns()
        logger.info("Pattern Recognition Engine cleaned up")
