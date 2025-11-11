"""
Performance Profiling and Bottleneck Identification
Provides tools to identify and analyze performance bottlenecks
"""

import time
import psutil
import cProfile
import pstats
import io
from functools import wraps
from typing import Dict, Any, List, Optional
from collections import defaultdict
import threading
from contextlib import contextmanager

class PerformanceProfiler:
    """Comprehensive performance profiling system"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.active_profiling = {}
        self.lock = threading.Lock()

    def start_operation(self, operation_name: str, metadata: Dict[str, Any] = None):
        """Start profiling an operation"""
        with self.lock:
            if operation_name in self.active_profiling:
                return  # Already profiling

            self.active_profiling[operation_name] = {
                'start_time': time.time(),
                'start_memory': psutil.Process().memory_info().rss,
                'start_cpu': psutil.cpu_percent(),
                'metadata': metadata or {},
                'thread_id': threading.get_ident()
            }

    def end_operation(self, operation_name: str) -> Dict[str, Any]:
        """End profiling an operation and return metrics"""
        with self.lock:
            if operation_name not in self.active_profiling:
                return {}

            start_data = self.active_profiling.pop(operation_name)

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            end_cpu = psutil.cpu_percent()

            metrics = {
                'operation': operation_name,
                'duration_ms': (end_time - start_data['start_time']) * 1000,
                'memory_delta_mb': (end_memory - start_data['start_memory']) / 1024 / 1024,
                'cpu_delta_percent': end_cpu - start_data['start_cpu'],
                'start_memory_mb': start_data['start_memory'] / 1024 / 1024,
                'end_memory_mb': end_memory / 1024 / 1024,
                'thread_id': start_data['thread_id'],
                'timestamp': end_time,
                **start_data['metadata']
            }

            # Store metrics for analysis
            self.metrics[operation_name].append(metrics)

            # Keep only last 1000 metrics per operation
            if len(self.metrics[operation_name]) > 1000:
                self.metrics[operation_name] = self.metrics[operation_name][-1000:]

            return metrics

    def get_operation_stats(self, operation_name: str, last_n: int = 100) -> Dict[str, Any]:
        """Get statistics for an operation"""
        with self.lock:
            if operation_name not in self.metrics:
                return {}

            operations = self.metrics[operation_name][-last_n:]
            if not operations:
                return {}

            durations = [op['duration_ms'] for op in operations]
            memory_deltas = [op['memory_delta_mb'] for op in operations]

            return {
                'operation': operation_name,
                'count': len(operations),
                'avg_duration_ms': sum(durations) / len(durations),
                'max_duration_ms': max(durations),
                'min_duration_ms': min(durations),
                'p95_duration_ms': sorted(durations)[int(len(durations) * 0.95)],
                'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas),
                'max_memory_delta_mb': max(memory_deltas),
                'total_memory_delta_mb': sum(memory_deltas)
            }

    def get_bottlenecks(self, threshold_ms: float = 1000) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        with self.lock:
            for operation_name, operations in self.metrics.items():
                if not operations:
                    continue

                recent_ops = operations[-100:]  # Last 100 operations
                slow_ops = [op for op in recent_ops if op['duration_ms'] > threshold_ms]

                if slow_ops:
                    bottlenecks.append({
                        'operation': operation_name,
                        'slow_operations_count': len(slow_ops),
                        'slow_operations_percentage': (len(slow_ops) / len(recent_ops)) * 100,
                        'avg_slow_duration_ms': sum(op['duration_ms'] for op in slow_ops) / len(slow_ops),
                        'max_slow_duration_ms': max(op['duration_ms'] for op in slow_ops),
                        'recent_total_operations': len(recent_ops)
                    })

        return sorted(bottlenecks, key=lambda x: x['slow_operations_percentage'], reverse=True)

    def profile_function(self, func):
        """Decorator to profile function performance"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f"{func.__module__}.{func.__name__}"
            metadata = {
                'args_count': len(args),
                'kwargs_count': len(kwargs),
                'function_name': func.__name__,
                'module_name': func.__module__
            }

            self.start_operation(operation_name, metadata)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                metrics = self.end_operation(operation_name)

                # Log slow operations
                if metrics.get('duration_ms', 0) > 1000:  # > 1 second
                    print(f"SLOW OPERATION: {operation_name} took {metrics['duration_ms']:.2f}ms")

        return wrapper

    def get_memory_usage_report(self) -> Dict[str, Any]:
        """Generate memory usage report"""
        process = psutil.Process()

        return {
            'rss_mb': process.memory_info().rss / 1024 / 1024,
            'vms_mb': process.memory_info().vms / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(),
            'threads_count': threading.active_count(),
            'open_files_count': len(process.open_files()),
            'connections_count': len(process.connections())
        }

    def clear_metrics(self, operation_name: str = None):
        """Clear stored metrics"""
        with self.lock:
            if operation_name:
                self.metrics[operation_name].clear()
            else:
                self.metrics.clear()

# Global profiler instance
profiler = PerformanceProfiler()

@contextmanager
def profile_operation(operation_name: str, **metadata):
    """Context manager for profiling operations"""
    profiler.start_operation(operation_name, metadata)
    try:
        yield
    finally:
        metrics = profiler.end_operation(operation_name)

        # Log if operation was slow
        if metrics.get('duration_ms', 0) > 500:  # > 500ms
            print(f"Profiled operation '{operation_name}': {metrics['duration_ms']:.2f}ms, "
                  f"Memory: {metrics['memory_delta_mb']:+.2f}MB")

def profile_function_detailed(func):
    """Detailed profiling decorator with cProfile"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            pr.disable()

            # Get profile stats
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(10)  # Top 10 functions

            duration = (end_time - start_time) * 1000
            memory_delta = (end_memory - start_memory) / 1024 / 1024

            if duration > 2000:  # Only log very slow operations
                print(f"DETAILED PROFILE for {func.__name__}:")
                print(f"Duration: {duration:.2f}ms")
                print(f"Memory delta: {memory_delta:.2f}MB")
                print("Top 10 functions by cumulative time:")
                print(s.getvalue())
                print("-" * 50)

    return wrapper

def get_performance_report() -> Dict[str, Any]:
    """Generate comprehensive performance report"""
    bottlenecks = profiler.get_bottlenecks()
    memory_report = profiler.get_memory_usage_report()

    # Get stats for key operations
    operation_stats = {}
    key_operations = [
        'query_processing', 'memory_search', 'vector_embedding',
        'neural_mesh_update', 'learning_pipeline'
    ]

    for op in key_operations:
        stats = profiler.get_operation_stats(op)
        if stats:
            operation_stats[op] = stats

    return {
        'bottlenecks': bottlenecks,
        'memory_usage': memory_report,
        'operation_stats': operation_stats,
        'timestamp': time.time(),
        'total_operations_tracked': sum(len(ops) for ops in profiler.metrics.values())
    }
