"""
Memory Metrics - Quality Assurance and Performance Tracking
Implements precision@k tracking, hallucination detection, and QA test harness
"""
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from .memory_manager import MemoryManager
from ..llm.memory_curator import MemoryCurator
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query operation"""
    query: str
    response: str
    memory_chunks_used: int
    total_chunks_found: int
    precision_at_k: Dict[int, float]
    hallucination_score: float
    response_relevance: float
    response_time_ms: float
    timestamp: float
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TestCase:
    """QA test case definition"""
    id: str
    query: str
    expected_answer: str
    expected_sources: List[str]
    category: str
    difficulty: str = "medium"  # easy, medium, hard

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryMetrics:
    """
    Comprehensive metrics collection and quality assurance system.
    Tracks precision@k, hallucination rates, and system performance.
    """

    def __init__(self, memory_manager: MemoryManager, curator: Optional[MemoryCurator] = None):
        """Initialize metrics system"""
        self.memory_manager = memory_manager
        self.curator = curator

        # Metrics storage
        self.query_history: List[QueryMetrics] = []
        self.performance_stats = {
            'total_queries': 0,
            'avg_response_time_ms': 0,
            'avg_precision_at_5': 0,
            'avg_hallucination_rate': 0,
            'total_hallucinations': 0,
            'cache_hit_rate': 0,
            'memory_efficiency': 1.0
        }

        # Quality thresholds
        self.quality_thresholds = {
            'min_precision_at_5': 0.6,
            'max_hallucination_rate': 0.15,
            'max_response_time_ms': 2000,
            'min_relevance_score': 0.7
        }

        # Test harness
        self.test_cases: List[TestCase] = []
        self.baseline_results: Dict[str, Any] = {}

        # Load existing metrics if available
        self._load_metrics()

        logger.info("Memory Metrics system initialized")

    def record_query_metrics(self, query: str, response: str,
                           memory_results: Dict[str, Any],
                           response_time_ms: float,
                           source: str = "api") -> QueryMetrics:
        """
        Record metrics for a query operation

        Args:
            query: The user's query
            response: The AI's response
            memory_results: Results from memory search
            response_time_ms: Response time in milliseconds
            source: Source of the query (api, test, etc.)

        Returns:
            QueryMetrics object with calculated metrics
        """
        # Extract memory chunks used
        memory_chunks = memory_results.get('results', [])
        memory_chunks_used = len(memory_chunks)
        total_chunks_found = memory_results.get('total_found', 0)

        # Calculate precision@k
        precision_at_k = self._calculate_precision_at_k(query, memory_chunks)

        # Detect hallucinations
        hallucination_score = self._detect_hallucinations(response, memory_chunks)

        # Calculate response relevance
        response_relevance = self._calculate_response_relevance(query, response, memory_chunks)

        # Create metrics object
        metrics = QueryMetrics(
            query=query,
            response=response,
            memory_chunks_used=memory_chunks_used,
            total_chunks_found=total_chunks_found,
            precision_at_k=precision_at_k,
            hallucination_score=hallucination_score,
            response_relevance=response_relevance,
            response_time_ms=response_time_ms,
            timestamp=time.time(),
            source=source
        )

        # Store metrics
        self.query_history.append(metrics)

        # Update rolling statistics
        self._update_performance_stats(metrics)

        # Keep only recent history (last 1000 queries)
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]

        logger.debug(f"Recorded metrics for query: {query[:50]}... (precision@5: {precision_at_k.get(5, 0):.2f})")

        return metrics

    def _calculate_precision_at_k(self, query: str, retrieved_chunks: List[Dict[str, Any]],
                                k_values: List[int] = [1, 3, 5, 10]) -> Dict[int, float]:
        """
        Calculate precision@k for retrieved chunks

        Args:
            query: The search query
            retrieved_chunks: Retrieved memory chunks
            k_values: Values of k to calculate

        Returns:
            Dictionary of precision@k scores
        """
        if not retrieved_chunks:
            return {k: 0.0 for k in k_values}

        # For now, use a simple heuristic: chunks with high scores are considered relevant
        # In a real system, this would use ground truth or human evaluation
        precision_scores = {}

        for k in k_values:
            if k > len(retrieved_chunks):
                k = len(retrieved_chunks)

            # Simple heuristic: consider top chunks as relevant based on score
            relevant_count = 0
            for i in range(k):
                chunk = retrieved_chunks[i]
                score = chunk.get('score', 0)

                # Heuristic: chunks with score > 0.7 are considered relevant
                if score > 0.7:
                    relevant_count += 1

            precision_scores[k] = relevant_count / k if k > 0 else 0.0

        return precision_scores

    def _detect_hallucinations(self, response: str, memory_chunks: List[Dict[str, Any]]) -> float:
        """
        Detect hallucinations in the response using curator model

        Args:
            response: AI response to check
            memory_chunks: Memory chunks used for the response

        Returns:
            Hallucination score (0.0 = no hallucinations, 1.0 = high hallucinations)
        """
        if not self.curator or not self.curator.is_ready():
            # Fallback: simple keyword-based detection
            return self._simple_hallucination_detection(response, memory_chunks)

        try:
            # Use curator model for sophisticated hallucination detection
            memory_text = "\n".join([chunk.get('text', '') for chunk in memory_chunks[:5]])

            prompt = f"""Analyze if this AI response contains hallucinations or unsupported claims.
Rate the hallucination level from 0.0 to 1.0 where:
0.0 = All claims are directly supported by the provided memory
1.0 = Response contains significant unsupported claims

Memory Context:
{memory_text}

AI Response:
{response}

Hallucination Score (just the number):"""

            result = self.curator.model_loader.generate_text(
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            ).strip()

            try:
                score = float(result)
                return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
            except ValueError:
                logger.warning(f"Could not parse hallucination score: {result}")
                return 0.5

        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return self._simple_hallucination_detection(response, memory_chunks)

    def _simple_hallucination_detection(self, response: str, memory_chunks: List[Dict[str, Any]]) -> float:
        """
        Simple keyword-based hallucination detection as fallback

        Args:
            response: AI response to check
            memory_chunks: Memory chunks used

        Returns:
            Simple hallucination score
        """
        # Extract key facts from memory
        memory_facts = set()
        for chunk in memory_chunks:
            text = chunk.get('text', '').lower()
            # Simple fact extraction (could be improved)
            words = text.split()
            memory_facts.update(words)

        # Check response against memory facts
        response_words = set(response.lower().split())
        novel_words = response_words - memory_facts

        # Calculate novelty ratio (higher = more potential hallucinations)
        if len(response_words) == 0:
            return 0.0

        novelty_ratio = len(novel_words) / len(response_words)

        # Scale to hallucination score (0-1)
        # Assume responses with >70% novel words might be hallucinatory
        hallucination_score = min(novelty_ratio / 0.7, 1.0)

        return hallucination_score

    def _calculate_response_relevance(self, query: str, response: str,
                                    memory_chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate how relevant the response is to the query

        Args:
            query: Original query
            response: AI response
            memory_chunks: Memory chunks used

        Returns:
            Relevance score (0.0 - 1.0)
        """
        if not self.curator or not self.curator.is_ready():
            # Simple fallback: check if query terms appear in response
            query_terms = set(query.lower().split())
            response_terms = set(response.lower().split())
            overlap = len(query_terms & response_terms)
            return overlap / len(query_terms) if query_terms else 0.0

        try:
            prompt = f"""Rate how relevant this response is to the query on a scale of 0.0 to 1.0.
0.0 = Completely irrelevant
1.0 = Directly and comprehensively answers the query

Query: {query}
Response: {response}

Relevance Score (just the number):"""

            result = self.curator.model_loader.generate_text(
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            ).strip()

            try:
                score = float(result)
                return min(max(score, 0.0), 1.0)
            except ValueError:
                return 0.5

        except Exception as e:
            logger.error(f"Relevance calculation failed: {e}")
            return 0.5

    def _update_performance_stats(self, metrics: QueryMetrics):
        """Update rolling performance statistics"""
        self.performance_stats['total_queries'] += 1

        # Update averages
        total_queries = self.performance_stats['total_queries']

        # Response time
        old_avg_time = self.performance_stats['avg_response_time_ms']
        self.performance_stats['avg_response_time_ms'] = (
            (old_avg_time * (total_queries - 1) + metrics.response_time_ms) / total_queries
        )

        # Precision@5
        old_avg_precision = self.performance_stats['avg_precision_at_5']
        precision_5 = metrics.precision_at_k.get(5, 0)
        self.performance_stats['avg_precision_at_5'] = (
            (old_avg_precision * (total_queries - 1) + precision_5) / total_queries
        )

        # Hallucination rate
        old_avg_hallucination = self.performance_stats['avg_hallucination_rate']
        self.performance_stats['avg_hallucination_rate'] = (
            (old_avg_hallucination * (total_queries - 1) + metrics.hallucination_score) / total_queries
        )

        # Track total hallucinations
        if metrics.hallucination_score > 0.5:
            self.performance_stats['total_hallucinations'] += 1

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        if not self.query_history:
            return {'error': 'No query history available'}

        recent_queries = [q for q in self.query_history
                         if time.time() - q.timestamp < 3600]  # Last hour

        report = {
            'overall_stats': self.performance_stats.copy(),
            'quality_assessment': {
                'precision_at_5_status': 'good' if self.performance_stats['avg_precision_at_5'] > 0.6 else 'needs_improvement',
                'hallucination_status': 'good' if self.performance_stats['avg_hallucination_rate'] < 0.15 else 'concerning',
                'response_time_status': 'good' if self.performance_stats['avg_response_time_ms'] < 2000 else 'slow'
            },
            'recent_performance': {
                'queries_last_hour': len(recent_queries),
                'avg_precision_recent': sum(q.precision_at_k.get(5, 0) for q in recent_queries) / len(recent_queries) if recent_queries else 0,
                'avg_hallucinations_recent': sum(q.hallucination_score for q in recent_queries) / len(recent_queries) if recent_queries else 0
            },
            'thresholds': self.quality_thresholds
        }

        return report

    def detect_regression(self, baseline_window_hours: int = 24) -> Dict[str, Any]:
        """
        Detect performance regression compared to baseline

        Args:
            baseline_window_hours: Hours to use for baseline calculation

        Returns:
            Regression analysis
        """
        current_time = time.time()
        baseline_cutoff = current_time - (baseline_window_hours * 3600)

        # Split queries into baseline and recent
        baseline_queries = [q for q in self.query_history if q.timestamp < baseline_cutoff]
        recent_queries = [q for q in self.query_history if q.timestamp >= baseline_cutoff]

        if len(baseline_queries) < 10 or len(recent_queries) < 5:
            return {'error': 'Insufficient data for regression analysis'}

        # Calculate baseline metrics
        baseline_precision = sum(q.precision_at_k.get(5, 0) for q in baseline_queries) / len(baseline_queries)
        baseline_hallucinations = sum(q.hallucination_score for q in baseline_queries) / len(baseline_queries)
        baseline_response_time = sum(q.response_time_ms for q in baseline_queries) / len(baseline_queries)

        # Calculate recent metrics
        recent_precision = sum(q.precision_at_k.get(5, 0) for q in recent_queries) / len(recent_queries)
        recent_hallucinations = sum(q.hallucination_score for q in recent_queries) / len(recent_queries)
        recent_response_time = sum(q.response_time_ms for q in recent_queries) / len(recent_queries)

        # Calculate changes
        precision_change = recent_precision - baseline_precision
        hallucination_change = recent_hallucinations - baseline_hallucinations
        response_time_change = recent_response_time - baseline_response_time

        regression_detected = (
            precision_change < -0.1 or  # 10% drop in precision
            hallucination_change > 0.1 or  # 10% increase in hallucinations
            response_time_change > 500  # 500ms increase in response time
        )

        return {
            'regression_detected': regression_detected,
            'baseline_period': {
                'queries': len(baseline_queries),
                'precision_at_5': baseline_precision,
                'hallucination_rate': baseline_hallucinations,
                'avg_response_time_ms': baseline_response_time
            },
            'recent_period': {
                'queries': len(recent_queries),
                'precision_at_5': recent_precision,
                'hallucination_rate': recent_hallucinations,
                'avg_response_time_ms': recent_response_time
            },
            'changes': {
                'precision_change': precision_change,
                'hallucination_change': hallucination_change,
                'response_time_change_ms': response_time_change
            }
        }

    def _load_metrics(self):
        """Load metrics from disk"""
        try:
            metrics_file = settings.data_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)

                # Load performance stats
                self.performance_stats.update(data.get('performance_stats', {}))

                # Load recent query history (last 100)
                history_data = data.get('query_history', [])
                for item in history_data[-100:]:
                    try:
                        metrics = QueryMetrics(**item)
                        self.query_history.append(metrics)
                    except:
                        continue

                logger.info(f"Loaded metrics data: {len(self.query_history)} queries")

        except Exception as e:
            logger.warning(f"Failed to load metrics: {e}")

    def _save_metrics(self):
        """Save metrics to disk"""
        try:
            metrics_file = settings.data_dir / "metrics.json"

            # Prepare data for serialization
            history_data = [q.to_dict() for q in self.query_history[-200:]]  # Save last 200

            data = {
                'performance_stats': self.performance_stats,
                'query_history': history_data,
                'last_updated': time.time()
            }

            with open(metrics_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def __del__(self):
        """Save metrics on destruction"""
        try:
            self._save_metrics()
        except:
            pass


class QATestHarness:
    """
    Quality Assurance test harness for automated testing of memory system.
    Runs test cases and compares results against expected outcomes.
    """

    def __init__(self, memory_manager: MemoryManager, metrics: MemoryMetrics,
                 test_dataset_path: Optional[Path] = None):
        """Initialize QA test harness"""
        self.memory_manager = memory_manager
        self.metrics = metrics

        # Test cases and results
        self.test_cases: List[TestCase] = []
        self.test_results: List[Dict[str, Any]] = []
        self.baseline_results: Dict[str, Any] = {}

        # Load test cases
        if test_dataset_path:
            self.load_test_cases(test_dataset_path)
        else:
            self._create_default_test_cases()

        logger.info(f"QA Test Harness initialized with {len(self.test_cases)} test cases")

    def load_test_cases(self, dataset_path: Path):
        """Load test cases from file"""
        try:
            if dataset_path.suffix == '.json':
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
                    for item in data.get('test_cases', []):
                        test_case = TestCase(**item)
                        self.test_cases.append(test_case)
            else:
                logger.warning(f"Unsupported test dataset format: {dataset_path.suffix}")

            logger.info(f"Loaded {len(self.test_cases)} test cases from {dataset_path}")

        except Exception as e:
            logger.error(f"Failed to load test cases: {e}")
            self._create_default_test_cases()

    def _create_default_test_cases(self):
        """Create default test cases for basic QA"""
        default_cases = [
            TestCase(
                id="basic_memory_recall",
                query="What is the capital of France?",
                expected_answer="Paris",
                expected_sources=[],
                category="general",
                difficulty="easy"
            ),
            TestCase(
                id="empty_memory_query",
                query="What is the meaning of life according to quantum physics?",
                expected_answer="I don't know",
                expected_sources=[],
                category="philosophy",
                difficulty="hard"
            ),
            TestCase(
                id="factual_accuracy_test",
                query="Who wrote Romeo and Juliet?",
                expected_answer="William Shakespeare",
                expected_sources=[],
                category="literature",
                difficulty="medium"
            ),
            TestCase(
                id="context_awareness_test",
                query="What programming language are we discussing?",
                expected_answer="Python",  # Assuming Python docs are ingested
                expected_sources=[],
                category="technical",
                difficulty="medium"
            ),
            TestCase(
                id="hallucination_detection",
                query="What is the current population of Mars?",
                expected_answer="I don't know",  # Should not hallucinate
                expected_sources=[],
                category="science",
                difficulty="hard"
            )
        ]

        self.test_cases.extend(default_cases)
        logger.info(f"Created {len(default_cases)} default test cases")

    def run_test_suite(self, llm_core=None, save_results: bool = True) -> Dict[str, Any]:
        """
        Run the complete test suite

        Args:
            llm_core: LLM core instance for end-to-end testing
            save_results: Whether to save results to disk

        Returns:
            Test suite results
        """
        logger.info(f"Running QA test suite with {len(self.test_cases)} test cases")

        results = []
        start_time = time.time()

        for test_case in self.test_cases:
            try:
                result = self.run_single_test(test_case, llm_core)
                results.append(result)
                self.test_results.append(result)

                logger.info(f"Test {test_case.id}: {result['status']} "
                          f"(precision@5: {result.get('precision_at_5', 0):.2f})")

            except Exception as e:
                logger.error(f"Test {test_case.id} failed: {e}")
                error_result = {
                    'test_id': test_case.id,
                    'status': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                }
                results.append(error_result)
                self.test_results.append(error_result)

        # Keep only recent results
        if len(self.test_results) > 500:
            self.test_results = self.test_results[-500:]

        # Calculate summary statistics
        summary = self._calculate_test_summary(results)

        total_time = time.time() - start_time
        summary['total_execution_time'] = total_time
        summary['tests_per_second'] = len(results) / total_time if total_time > 0 else 0

        # Save results if requested
        if save_results:
            self._save_test_results(results, summary)

        logger.info(f"Test suite completed: {summary}")
        return summary

    def run_single_test(self, test_case: TestCase, llm_core=None) -> Dict[str, Any]:
        """
        Run a single test case

        Args:
            test_case: Test case to run
            llm_core: LLM core for end-to-end testing

        Returns:
            Test result
        """
        result = {
            'test_id': test_case.id,
            'query': test_case.query,
            'expected_answer': test_case.expected_answer,
            'timestamp': time.time(),
            'status': 'unknown'
        }

        try:
            # Test memory retrieval
            memory_results = self.memory_manager.hybrid_search_cached(
                test_case.query,
                category=test_case.category,
                max_results=5
            )

            result['memory_results'] = memory_results
            result['chunks_found'] = len(memory_results.get('results', []))

            # Calculate precision@k
            precision_scores = self.metrics._calculate_precision_at_k(
                test_case.query, memory_results.get('results', [])
            )
            result['precision_at_k'] = precision_scores
            result['precision_at_5'] = precision_scores.get(5, 0)

            # If LLM core is available, test end-to-end response
            if llm_core:
                response_start = time.time()
                llm_response = llm_core.query(
                    test_case.query,
                    category=test_case.category,
                    temperature=0.1  # Low temperature for consistent results
                )
                response_time = time.time() - response_start

                result['llm_response'] = llm_response.get('answer', '')
                result['response_time_ms'] = response_time * 1000

                # Evaluate response quality
                hallucination_score = self.metrics._detect_hallucinations(
                    result['llm_response'], memory_results.get('results', [])
                )
                relevance_score = self.metrics._calculate_response_relevance(
                    test_case.query, result['llm_response'], memory_results.get('results', [])
                )

                result['hallucination_score'] = hallucination_score
                result['relevance_score'] = relevance_score

                # Simple answer matching (could be improved with NLP)
                expected_lower = test_case.expected_answer.lower().strip()
                response_lower = result['llm_response'].lower().strip()

                # Check for key phrases
                answer_correct = (
                    expected_lower in response_lower or
                    any(phrase in response_lower for phrase in expected_lower.split())
                )
                result['answer_correct'] = answer_correct

            # Determine test status
            if result.get('precision_at_5', 0) > 0.6:  # Good memory retrieval
                if llm_core and result.get('hallucination_score', 0) < 0.3:  # Low hallucinations
                    result['status'] = 'pass'
                elif not llm_core:
                    result['status'] = 'pass'  # Memory-only test
                else:
                    result['status'] = 'degraded'
            else:
                result['status'] = 'fail'

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            logger.error(f"Test {test_case.id} execution failed: {e}")

        return result

    def _calculate_test_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from test results"""
        if not results:
            return {'error': 'No test results'}

        total_tests = len(results)
        passed_tests = len([r for r in results if r.get('status') == 'pass'])
        failed_tests = len([r for r in results if r.get('status') == 'fail'])
        error_tests = len([r for r in results if r.get('status') == 'error'])
        degraded_tests = len([r for r in results if r.get('status') == 'degraded'])

        # Calculate averages
        precision_scores = [r.get('precision_at_5', 0) for r in results if 'precision_at_5' in r]
        hallucination_scores = [r.get('hallucination_score', 0) for r in results if 'hallucination_score' in r]
        relevance_scores = [r.get('relevance_score', 0) for r in results if 'relevance_score' in r]

        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        avg_hallucinations = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0

        # Calculate pass rate
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0

        summary = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'errors': error_tests,
            'degraded': degraded_tests,
            'pass_rate': pass_rate,
            'average_precision_at_5': avg_precision,
            'average_hallucination_rate': avg_hallucinations,
            'average_relevance_score': avg_relevance,
            'quality_assessment': self._assess_quality(avg_precision, avg_hallucinations, pass_rate)
        }

        return summary

    def _assess_quality(self, avg_precision: float, avg_hallucinations: float,
                       pass_rate: float) -> str:
        """Assess overall quality based on metrics"""
        if avg_precision > 0.7 and avg_hallucinations < 0.1 and pass_rate > 0.8:
            return 'excellent'
        elif avg_precision > 0.6 and avg_hallucinations < 0.15 and pass_rate > 0.7:
            return 'good'
        elif avg_precision > 0.5 and avg_hallucinations < 0.2 and pass_rate > 0.6:
            return 'acceptable'
        elif avg_precision > 0.4 or avg_hallucinations > 0.3 or pass_rate < 0.5:
            return 'poor'
        else:
            return 'needs_improvement'

    def establish_baseline(self, llm_core=None) -> Dict[str, Any]:
        """
        Establish baseline performance metrics

        Args:
            llm_core: LLM core for end-to-end testing

        Returns:
            Baseline results
        """
        logger.info("Establishing performance baseline...")

        results = self.run_test_suite(llm_core, save_results=False)
        self.baseline_results = {
            'timestamp': time.time(),
            'results': results,
            'test_cases': [tc.to_dict() for tc in self.test_cases]
        }

        # Save baseline
        try:
            baseline_file = settings.data_dir / "baseline_results.json"
            with open(baseline_file, 'w') as f:
                json.dump(self.baseline_results, f, indent=2, default=str)
            logger.info("Baseline results saved")
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")

        return results

    def compare_to_baseline(self) -> Dict[str, Any]:
        """
        Compare current performance to established baseline

        Returns:
            Comparison results
        """
        if not self.baseline_results:
            return {'error': 'No baseline established'}

        # Run current tests
        current_results = self.run_test_suite(save_results=False)

        baseline = self.baseline_results['results']

        # Compare key metrics
        comparison = {
            'baseline_timestamp': self.baseline_results['timestamp'],
            'current_timestamp': time.time(),
            'baseline_metrics': baseline,
            'current_metrics': current_results,
            'changes': {}
        }

        # Calculate changes
        for metric in ['pass_rate', 'average_precision_at_5', 'average_hallucination_rate', 'average_relevance_score']:
            baseline_value = baseline.get(metric, 0)
            current_value = current_results.get(metric, 0)
            change = current_value - baseline_value
            comparison['changes'][metric] = {
                'baseline': baseline_value,
                'current': current_value,
                'change': change,
                'change_percent': (change / baseline_value * 100) if baseline_value != 0 else 0
            }

        # Determine if performance has regressed
        significant_regression = (
            comparison['changes'].get('pass_rate', {}).get('change', 0) < -0.1 or
            comparison['changes'].get('average_precision_at_5', {}).get('change', 0) < -0.1 or
            comparison['changes'].get('average_hallucination_rate', {}).get('change', 0) > 0.1
        )

        comparison['regression_detected'] = significant_regression

        logger.info(f"Baseline comparison: regression_detected={significant_regression}")
        return comparison

    def _save_test_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any]):
        """Save test results to disk"""
        try:
            results_file = settings.data_dir / "qa_test_results.json"

            data = {
                'timestamp': time.time(),
                'summary': summary,
                'results': results[-100:],  # Save last 100 detailed results
                'test_suite_size': len(self.test_cases)
            }

            with open(results_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Test results saved to {results_file}")

        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

    def get_test_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent test run history"""
        try:
            results_file = settings.data_dir / "qa_test_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    return [data]  # Return as list for consistency
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to load test history: {e}")
            return []

    def add_test_case(self, test_case: TestCase):
        """Add a new test case to the suite"""
        self.test_cases.append(test_case)
        logger.info(f"Added test case: {test_case.id}")

    def remove_test_case(self, test_id: str) -> bool:
        """Remove a test case from the suite"""
        original_length = len(self.test_cases)
        self.test_cases = [tc for tc in self.test_cases if tc.id != test_id]

        if len(self.test_cases) < original_length:
            logger.info(f"Removed test case: {test_id}")
            return True
        else:
            logger.warning(f"Test case not found: {test_id}")
            return False
