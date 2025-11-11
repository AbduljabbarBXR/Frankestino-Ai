"""
Autonomy System for Frankenstino AI
Phase 3: Autonomy Transition - Critical mass detection, hybrid operation modes, and self-learning enhancements
"""
import logging
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

from ..config import settings

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels for routing decisions"""
    SIMPLE = "simple"  # Factual, direct questions
    MODERATE = "moderate"  # Requires some reasoning
    COMPLEX = "complex"  # Needs deep reasoning or multi-step logic


class OperationMode(Enum):
    """Available operation modes"""
    HYBRID = "hybrid"  # Traditional memory-augmented + LLM
    SUBSTRATE_ONLY = "substrate_only"  # Neural network only
    ADAPTIVE = "adaptive"  # Choose based on maturity and query


@dataclass
class MaturityMetrics:
    """Comprehensive maturity assessment metrics"""
    # Structural metrics
    node_count: int = 0
    edge_count: int = 0
    connection_density: float = 0.0

    # Semantic quality metrics
    semantic_relationship_score: float = 0.0
    relationship_type_diversity: float = 0.0
    confidence_score_avg: float = 0.0

    # Learning effectiveness metrics
    pattern_learning_score: float = 0.0
    hebbian_learning_effectiveness: float = 0.0
    query_success_rate: float = 0.0

    # Network coherence metrics
    cluster_quality_score: float = 0.0
    information_flow_efficiency: float = 0.0
    temporal_stability: float = 0.0

    # Performance metrics
    substrate_query_speed: float = 0.0
    hybrid_query_speed: float = 0.0
    substrate_confidence_avg: float = 0.0

    # Overall maturity score
    overall_maturity: float = 0.0
    autonomous_ready: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryAnalysis:
    """Analysis of query characteristics for routing"""
    complexity: QueryComplexity
    estimated_difficulty: float  # 0.0 to 1.0
    requires_reasoning: bool
    is_factual: bool
    has_temporal_aspects: bool
    involves_relationships: bool
    confidence_threshold: float


@dataclass
class LearningFeedback:
    """Feedback from query processing for self-learning"""
    query: str
    mode_used: OperationMode
    success: bool
    confidence: float
    processing_time: float
    answer_quality: float  # 0.0 to 1.0
    timestamp: float


class MaturityAssessor:
    """Enhanced maturity assessment for autonomy transition"""

    def __init__(self, neural_mesh, memory_manager, signal_processor):
        self.neural_mesh = neural_mesh
        self.memory_manager = memory_manager
        self.signal_processor = signal_processor

        # Historical data for trend analysis
        self.maturity_history: List[Tuple[float, MaturityMetrics]] = []
        self.query_feedback_history: List[LearningFeedback] = []

        # Self-learning parameters (will be optimized over time)
        self.learning_params = {
            'semantic_weight': 0.25,
            'learning_weight': 0.25,
            'coherence_weight': 0.25,
            'performance_weight': 0.25
        }

    def assess_maturity(self) -> MaturityMetrics:
        """Comprehensive maturity assessment"""
        metrics = MaturityMetrics()

        # Structural metrics
        metrics.node_count = len(self.neural_mesh.nodes)
        metrics.edge_count = len(self.neural_mesh.edges)
        metrics.connection_density = metrics.edge_count / max(metrics.node_count, 1)

        # Semantic quality metrics
        metrics.semantic_relationship_score = self._calculate_semantic_quality()
        metrics.relationship_type_diversity = self._calculate_relationship_diversity()
        metrics.confidence_score_avg = self._calculate_avg_confidence()

        # Learning effectiveness metrics
        metrics.pattern_learning_score = self._calculate_pattern_learning_score()
        metrics.hebbian_learning_effectiveness = self._calculate_hebbian_effectiveness()
        metrics.query_success_rate = self._calculate_query_success_rate()

        # Network coherence metrics
        metrics.cluster_quality_score = self._calculate_cluster_quality()
        metrics.information_flow_efficiency = self._calculate_information_flow()
        metrics.temporal_stability = self._calculate_temporal_stability()

        # Performance metrics
        perf_stats = self._calculate_performance_metrics()
        metrics.substrate_query_speed = perf_stats['substrate_speed']
        metrics.hybrid_query_speed = perf_stats['hybrid_speed']
        metrics.substrate_confidence_avg = perf_stats['substrate_confidence']

        # Calculate overall maturity score
        metrics.overall_maturity = self._calculate_overall_maturity(metrics)
        metrics.autonomous_ready = metrics.overall_maturity >= settings.autonomy_maturity_threshold

        # Store in history for trend analysis
        self.maturity_history.append((time.time(), metrics))

        # Keep only last 100 assessments
        if len(self.maturity_history) > 100:
            self.maturity_history = self.maturity_history[-100:]

        return metrics

    def _calculate_semantic_quality(self) -> float:
        """Calculate semantic relationship quality score"""
        if not self.neural_mesh.edges:
            return 0.0

        total_quality = 0.0
        count = 0

        for edge in self.neural_mesh.edges.values():
            # Quality based on relationship type specificity and confidence
            relationship_type = edge.metadata.get('relationship_type', '')
            confidence = edge.metadata.get('confidence', 0.5)

            # More specific relationship types get higher scores
            type_score = 1.0 if relationship_type else 0.5
            type_score *= (1.0 if len(relationship_type.split('_')) > 1 else 0.7)

            quality = type_score * confidence
            total_quality += quality
            count += 1

        return total_quality / max(count, 1)

    def _calculate_relationship_diversity(self) -> float:
        """Calculate diversity of relationship types"""
        relationship_types = set()

        for edge in self.neural_mesh.edges.values():
            rel_type = edge.metadata.get('relationship_type', '')
            if rel_type:
                relationship_types.add(rel_type)

        # Normalize to 0-1 scale (more types = higher diversity)
        diversity = min(len(relationship_types) / 10.0, 1.0)  # Cap at 10 types
        return diversity

    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence in relationships"""
        if not self.neural_mesh.edges:
            return 0.0

        confidences = [edge.metadata.get('confidence', 0.5)
                      for edge in self.neural_mesh.edges.values()]
        return statistics.mean(confidences) if confidences else 0.0

    def _calculate_pattern_learning_score(self) -> float:
        """Calculate effectiveness of pattern learning"""
        # This would analyze pattern recognition engine performance
        # For now, return a placeholder based on pattern count
        pattern_count = getattr(self.neural_mesh, 'learned_patterns', {}).get('count', 0)
        return min(pattern_count / 100.0, 1.0)  # Scale based on pattern count

    def _calculate_hebbian_effectiveness(self) -> float:
        """Calculate Hebbian learning effectiveness"""
        if not self.neural_mesh.edges:
            return 0.0

        # Measure reinforcement distribution
        reinforcements = [edge.reinforcement_count for edge in self.neural_mesh.edges.values()]
        if not reinforcements:
            return 0.0

        # Effectiveness based on reinforcement variance (active learning)
        try:
            variance = statistics.variance(reinforcements)
            # Higher variance indicates active differential learning
            effectiveness = min(variance / 10.0, 1.0)
            return effectiveness
        except statistics.StatisticsError:
            return 0.0

    def _calculate_query_success_rate(self) -> float:
        """Calculate substrate query success rate"""
        if not self.query_feedback_history:
            return 0.0

        recent_feedback = self.query_feedback_history[-50:]  # Last 50 queries
        substrate_queries = [f for f in recent_feedback if f.mode_used == OperationMode.SUBSTRATE_ONLY]

        if not substrate_queries:
            return 0.0

        success_count = sum(1 for f in substrate_queries if f.success)
        return success_count / len(substrate_queries)

    def _calculate_cluster_quality(self) -> float:
        """Calculate neural network cluster quality"""
        # Analyze connected components and their coherence
        if not self.neural_mesh.nodes:
            return 0.0

        # Simple cluster analysis based on connection patterns
        connected_nodes = set()
        for node_id in self.neural_mesh.nodes:
            if self.neural_mesh.adjacency_list.get(node_id):
                connected_nodes.add(node_id)

        connectivity_ratio = len(connected_nodes) / len(self.neural_mesh.nodes)
        return connectivity_ratio

    def _calculate_information_flow(self) -> float:
        """Calculate information flow efficiency"""
        # Measure how well information propagates through the network
        if not self.neural_mesh.nodes or not self.neural_mesh.edges:
            return 0.0

        # Calculate average shortest path or propagation efficiency
        # Simplified: based on network diameter approximation
        avg_degree = sum(len(connections) for connections in self.neural_mesh.adjacency_list.values()) / len(self.neural_mesh.nodes)
        efficiency = min(avg_degree / 5.0, 1.0)  # Normalize to 0-1
        return efficiency

    def _calculate_temporal_stability(self) -> float:
        """Calculate temporal stability of the network"""
        if len(self.maturity_history) < 2:
            return 0.5  # Default stability

        # Compare recent maturity scores
        recent_scores = [m.overall_maturity for _, m in self.maturity_history[-10:]]
        if len(recent_scores) < 2:
            return 0.5

        # Stability based on score variance (lower variance = higher stability)
        try:
            variance = statistics.variance(recent_scores)
            stability = max(0.0, 1.0 - variance)  # Invert variance for stability
            return stability
        except statistics.StatisticsError:
            return 0.5

    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance comparison metrics"""
        substrate_times = []
        hybrid_times = []
        substrate_confidences = []

        for feedback in self.query_feedback_history[-20:]:  # Last 20 queries
            if feedback.mode_used == OperationMode.SUBSTRATE_ONLY:
                substrate_times.append(feedback.processing_time)
                substrate_confidences.append(feedback.confidence)
            elif feedback.mode_used == OperationMode.HYBRID:
                hybrid_times.append(feedback.processing_time)

        return {
            'substrate_speed': statistics.mean(substrate_times) if substrate_times else float('inf'),
            'hybrid_speed': statistics.mean(hybrid_times) if hybrid_times else float('inf'),
            'substrate_confidence': statistics.mean(substrate_confidences) if substrate_confidences else 0.0
        }

    def _calculate_overall_maturity(self, metrics: MaturityMetrics) -> float:
        """Calculate overall maturity score using weighted combination"""
        weights = self.learning_params

        semantic_score = (
            metrics.semantic_relationship_score * 0.4 +
            metrics.relationship_type_diversity * 0.3 +
            metrics.confidence_score_avg * 0.3
        )

        learning_score = (
            metrics.pattern_learning_score * 0.4 +
            metrics.hebbian_learning_effectiveness * 0.4 +
            metrics.query_success_rate * 0.2
        )

        coherence_score = (
            metrics.cluster_quality_score * 0.4 +
            metrics.information_flow_efficiency * 0.4 +
            metrics.temporal_stability * 0.2
        )

        performance_score = (
            min(metrics.substrate_query_speed / 2.0, 1.0) * 0.5 +  # Speed (lower is better)
            metrics.substrate_confidence_avg * 0.5
        )

        overall = (
            semantic_score * weights['semantic_weight'] +
            learning_score * weights['learning_weight'] +
            coherence_score * weights['coherence_weight'] +
            performance_score * weights['performance_weight']
        )

        return min(max(overall, 0.0), 1.0)

    def add_query_feedback(self, feedback: LearningFeedback):
        """Add query feedback for learning"""
        self.query_feedback_history.append(feedback)

        # Keep only recent feedback
        if len(self.query_feedback_history) > 1000:
            self.query_feedback_history = self.query_feedback_history[-1000:]

        # Trigger parameter optimization periodically
        if len(self.query_feedback_history) % 50 == 0:
            self._optimize_learning_parameters()

    def _optimize_learning_parameters(self):
        """Self-optimize learning parameters based on performance"""
        # Simple hill-climbing optimization
        # This could be made more sophisticated with proper optimization algorithms

        current_performance = self._evaluate_parameter_performance(self.learning_params)

        # Try small adjustments to each parameter
        best_params = self.learning_params.copy()
        best_performance = current_performance

        for param in self.learning_params:
            for delta in [-0.05, 0.05]:
                test_params = self.learning_params.copy()
                test_params[param] = max(0.0, min(1.0, test_params[param] + delta))

                # Normalize weights to sum to 1.0
                total = sum(test_params.values())
                test_params = {k: v/total for k, v in test_params.items()}

                performance = self._evaluate_parameter_performance(test_params)
                if performance > best_performance:
                    best_performance = performance
                    best_params = test_params

        self.learning_params = best_params
        logger.info(f"Optimized learning parameters: {self.learning_params}")

    def _evaluate_parameter_performance(self, params: Dict[str, float]) -> float:
        """Evaluate how well a set of parameters performs"""
        if len(self.maturity_history) < 5:
            return 0.5  # Not enough data

        # Calculate recent maturity trend with these parameters
        recent_maturities = []
        for timestamp, metrics in self.maturity_history[-5:]:
            # Recalculate maturity with test parameters
            test_maturity = (
                (metrics.semantic_relationship_score * 0.4 +
                 metrics.relationship_type_diversity * 0.3 +
                 metrics.confidence_score_avg * 0.3) * params['semantic_weight'] +

                (metrics.pattern_learning_score * 0.4 +
                 metrics.hebbian_learning_effectiveness * 0.4 +
                 metrics.query_success_rate * 0.2) * params['learning_weight'] +

                (metrics.cluster_quality_score * 0.4 +
                 metrics.information_flow_efficiency * 0.4 +
                 metrics.temporal_stability * 0.2) * params['coherence_weight'] +

                (min(metrics.substrate_query_speed / 2.0, 1.0) * 0.5 +
                 metrics.substrate_confidence_avg * 0.5) * params['performance_weight']
            )
            recent_maturities.append(min(max(test_maturity, 0.0), 1.0))

        # Performance based on maturity improvement trend
        if len(recent_maturities) >= 2:
            trend = recent_maturities[-1] - recent_maturities[0]
            return max(0.0, trend + 0.5)  # Center around 0.5
        return 0.5


class QueryAnalyzer:
    """Analyzes queries to determine optimal processing mode"""

    def __init__(self, llm_interface=None):
        self.llm_interface = llm_interface

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query characteristics"""
        query_lower = query.lower()

        # Simple heuristics for complexity analysis
        complexity_indicators = {
            'simple': ['what is', 'who is', 'when did', 'where is', 'how many', 'list'],
            'complex': ['explain why', 'analyze', 'compare', 'evaluate', 'design', 'create'],
            'reasoning': ['because', 'therefore', 'however', 'although', 'why', 'how']
        }

        # Count complexity indicators
        simple_count = sum(1 for word in complexity_indicators['simple'] if word in query_lower)
        complex_count = sum(1 for word in complexity_indicators['complex'] if word in query_lower)
        reasoning_count = sum(1 for word in complexity_indicators['reasoning'] if word in query_lower)

        # Determine complexity
        if simple_count > complex_count and reasoning_count < 2:
            complexity = QueryComplexity.SIMPLE
            estimated_difficulty = 0.2
        elif complex_count > simple_count or reasoning_count >= 2:
            complexity = QueryComplexity.COMPLEX
            estimated_difficulty = 0.8
        else:
            complexity = QueryComplexity.MODERATE
            estimated_difficulty = 0.5

        # Additional analysis
        requires_reasoning = reasoning_count > 0 or complexity == QueryComplexity.COMPLEX
        is_factual = any(word in query_lower for word in ['what is', 'who is', 'when', 'where', 'how many'])
        has_temporal_aspects = any(word in query_lower for word in ['when', 'before', 'after', 'during', 'since', 'until'])
        involves_relationships = any(word in query_lower for word in ['relationship', 'connected', 'related', 'between'])

        # Confidence threshold based on complexity
        confidence_threshold = 0.3 if complexity == QueryComplexity.SIMPLE else 0.6

        return QueryAnalysis(
            complexity=complexity,
            estimated_difficulty=estimated_difficulty,
            requires_reasoning=requires_reasoning,
            is_factual=is_factual,
            has_temporal_aspects=has_temporal_aspects,
            involves_relationships=involves_relationships,
            confidence_threshold=confidence_threshold
        )


class HybridRouter:
    """Intelligent router for hybrid operation modes"""

    def __init__(self, maturity_assessor: MaturityAssessor, query_analyzer: QueryAnalyzer):
        self.maturity_assessor = maturity_assessor
        self.query_analyzer = query_analyzer

        # Routing statistics
        self.routing_stats = {
            'total_queries': 0,
            'substrate_only': 0,
            'hybrid': 0,
            'adaptive_switches': 0
        }

    def determine_operation_mode(self, query: str, context: Dict[str, Any] = None) -> OperationMode:
        """Determine the best operation mode for a query"""
        self.routing_stats['total_queries'] += 1

        # Get current maturity
        maturity = self.maturity_assessor.assess_maturity()

        # Analyze query
        query_analysis = self.query_analyzer.analyze_query(query)

        # Routing logic based on maturity and query characteristics
        if not maturity.autonomous_ready:
            # Not mature enough - use hybrid
            mode = OperationMode.HYBRID
            self.routing_stats['hybrid'] += 1

        elif settings.substrate_only_simple_queries and query_analysis.complexity == QueryComplexity.SIMPLE:
            # Simple queries can use substrate-only when enabled
            mode = OperationMode.SUBSTRATE_ONLY
            self.routing_stats['substrate_only'] += 1

        elif maturity.overall_maturity > 0.8 and query_analysis.estimated_difficulty < 0.6:
            # High maturity and moderate difficulty - try substrate-only
            mode = OperationMode.SUBSTRATE_ONLY
            self.routing_stats['substrate_only'] += 1

        else:
            # Default to hybrid for complex queries or when unsure
            mode = OperationMode.HYBRID
            self.routing_stats['hybrid'] += 1

        return mode

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return self.routing_stats.copy()


class SelfLearningSystem:
    """Self-learning enhancements for continuous improvement"""

    def __init__(self, maturity_assessor: MaturityAssessor, hybrid_router: HybridRouter):
        self.maturity_assessor = maturity_assessor
        self.hybrid_router = hybrid_router

        # Learning state
        self.learning_state = {
            'parameter_optimization_enabled': True,
            'confidence_calibration_active': True,
            'pattern_adaptation_enabled': True,
            'last_optimization': 0,
            'optimization_interval': 3600  # 1 hour
        }

    def process_feedback(self, feedback: LearningFeedback):
        """Process feedback for self-learning"""
        # Add to maturity assessor for parameter optimization
        self.maturity_assessor.add_query_feedback(feedback)

        # Adapt routing based on feedback
        self._adapt_routing_strategy(feedback)

        # Update confidence thresholds if needed
        self._update_confidence_thresholds(feedback)

    def _adapt_routing_strategy(self, feedback: LearningFeedback):
        """Adapt routing strategy based on feedback"""
        # If substrate-only failed but hybrid succeeded, be more conservative
        if (feedback.mode_used == OperationMode.SUBSTRATE_ONLY and
            not feedback.success and
            feedback.confidence < 0.5):
            # Could adjust routing parameters here
            logger.debug("Substrate query failed, adjusting routing conservatism")

    def _update_confidence_thresholds(self, feedback: LearningFeedback):
        """Update confidence thresholds based on performance"""
        # Dynamic threshold adjustment based on recent performance
        recent_feedback = self.maturity_assessor.query_feedback_history[-20:]

        if len(recent_feedback) >= 10:
            substrate_feedback = [f for f in recent_feedback if f.mode_used == OperationMode.SUBSTRATE_ONLY]

            if substrate_feedback:
                avg_confidence = statistics.mean(f.confidence for f in substrate_feedback)
                success_rate = sum(1 for f in substrate_feedback if f.success) / len(substrate_feedback)

                # Adjust threshold based on performance
                if success_rate > 0.8 and avg_confidence > 0.7:
                    # Performing well, can be less conservative
                    logger.debug("High substrate performance, adjusting thresholds")
                elif success_rate < 0.6:
                    # Performing poorly, be more conservative
                    logger.debug("Low substrate performance, increasing conservatism")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get self-learning statistics"""
        return {
            'learning_state': self.learning_state.copy(),
            'maturity_trends': len(self.maturity_assessor.maturity_history),
            'feedback_count': len(self.maturity_assessor.query_feedback_history),
            'parameter_optimization': self.learning_state['parameter_optimization_enabled']
        }


class AutonomySystem:
    """Main autonomy system coordinating all Phase 3 components"""

    def __init__(self, neural_mesh, memory_manager, signal_processor, llm_core):
        self.neural_mesh = neural_mesh
        self.memory_manager = memory_manager
        self.signal_processor = signal_processor
        self.llm_core = llm_core

        # Initialize components
        self.maturity_assessor = MaturityAssessor(neural_mesh, memory_manager, signal_processor)
        self.query_analyzer = QueryAnalyzer(llm_core)
        self.hybrid_router = HybridRouter(self.maturity_assessor, self.query_analyzer)
        self.self_learning = SelfLearningSystem(self.maturity_assessor, self.hybrid_router)

        logger.info("Autonomy System initialized for Phase 3")

    def assess_maturity(self) -> MaturityMetrics:
        """Get current maturity assessment"""
        return self.maturity_assessor.assess_maturity()

    def route_query(self, query: str, context: Dict[str, Any] = None) -> OperationMode:
        """Determine optimal operation mode for query"""
        return self.hybrid_router.determine_operation_mode(query, context)

    def process_query_with_feedback(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process query with autonomy system and collect feedback"""
        start_time = time.time()

        # Determine operation mode
        mode = self.route_query(query, context)

        # Process query
        if mode == OperationMode.SUBSTRATE_ONLY:
            result = self.llm_core.query_substrate(query, context)
            confidence = result.get('confidence', 0.0)
            success = result.get('method') != 'substrate_failed'
        else:  # HYBRID
            result = self.llm_core.query(query, context=context)
            confidence = 0.8  # Assume hybrid is generally reliable
            success = 'error' not in result

        processing_time = time.time() - start_time

        # Create feedback
        feedback = LearningFeedback(
            query=query,
            mode_used=mode,
            success=success,
            confidence=confidence,
            processing_time=processing_time,
            answer_quality=self._estimate_answer_quality(result),
            timestamp=time.time()
        )

        # Process feedback for learning
        self.self_learning.process_feedback(feedback)

        # Add autonomy metadata
        result['autonomy'] = {
            'mode_used': mode.value,
            'maturity_score': self.maturity_assessor.assess_maturity().overall_maturity,
            'confidence': confidence,
            'processing_time': processing_time
        }

        return result

    def _estimate_answer_quality(self, result: Dict[str, Any]) -> float:
        """Estimate answer quality for feedback"""
        # Simple heuristics for quality estimation
        answer = result.get('answer', '')

        if not answer or len(answer.strip()) < 10:
            return 0.1  # Too short

        if 'I don\'t know' in answer or 'error' in answer.lower():
            return 0.2  # Unhelpful

        if 'substrate' in result.get('method', '') and result.get('confidence', 0) > 0.7:
            return 0.8  # Good substrate result

        return 0.6  # Default moderate quality

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive autonomy system statistics"""
        maturity = self.maturity_assessor.assess_maturity()

        return {
            'maturity': maturity.to_dict(),
            'routing': self.hybrid_router.get_routing_stats(),
            'learning': self.self_learning.get_learning_stats(),
            'phase': 'autonomy_transition',
            'autonomous_ready': maturity.autonomous_ready
        }
