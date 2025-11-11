"""
Enhanced Error Handling and Logging System
Provides structured error handling, logging, and recovery strategies
"""

import logging
import time
import psutil
from contextlib import contextmanager
from functools import wraps
from typing import Dict, Any, Optional, Callable
from enum import Enum

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    NETWORK = "network"
    MEMORY = "memory"
    VALIDATION = "validation"
    PROCESSING = "processing"
    EXTERNAL = "external"
    INTERNAL = "internal"

class ErrorHandler:
    """Centralized error handling and logging system"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def classify_error(error: Exception) -> ErrorCategory:
        """Classify error by type for appropriate handling"""
        error_type = type(error).__name__

        # Network-related errors
        if any(err in error_type.lower() for err in ['connection', 'timeout', 'http', 'network']):
            return ErrorCategory.NETWORK

        # Memory-related errors
        elif any(err in error_type.lower() for err in ['memory', 'outofmemory', 'recursion']):
            return ErrorCategory.MEMORY

        # Validation errors
        elif any(err in error_type.lower() for err in ['value', 'type', 'attribute', 'key']):
            return ErrorCategory.VALIDATION

        # Processing errors
        elif any(err in error_type.lower() for err in ['processing', 'inference', 'embedding']):
            return ErrorCategory.PROCESSING

        # External service errors
        elif any(err in error_type.lower() for err in ['external', 'service', 'api']):
            return ErrorCategory.EXTERNAL

        # Default to internal
        else:
            return ErrorCategory.INTERNAL

    @staticmethod
    def get_error_severity(error: Exception, context: Dict[str, Any] = None) -> ErrorSeverity:
        """Determine error severity based on error type and context"""
        error_category = ErrorHandler.classify_error(error)

        # Critical errors
        if error_category in [ErrorCategory.MEMORY, ErrorCategory.EXTERNAL]:
            return ErrorSeverity.CRITICAL

        # High severity
        if error_category == ErrorCategory.PROCESSING:
            return ErrorSeverity.HIGH

        # Medium severity
        if error_category == ErrorCategory.NETWORK:
            return ErrorSeverity.MEDIUM

        # Low severity
        return ErrorSeverity.LOW

    @staticmethod
    def get_recovery_strategy(error_category: ErrorCategory) -> Optional[Callable]:
        """Get appropriate recovery strategy for error category"""
        strategies = {
            ErrorCategory.NETWORK: lambda: time.sleep(1),  # Retry after delay
            ErrorCategory.MEMORY: lambda: psutil.Process().memory_info(),  # Log memory usage
            ErrorCategory.VALIDATION: lambda: None,  # Log and continue
            ErrorCategory.PROCESSING: lambda: time.sleep(0.1),  # Brief pause
            ErrorCategory.EXTERNAL: lambda: time.sleep(2),  # Longer retry
            ErrorCategory.INTERNAL: lambda: None  # Log only
        }
        return strategies.get(error_category)

    def log_error(self, error: Exception, context: Dict[str, Any] = None,
                  operation: str = "unknown", severity: ErrorSeverity = None):
        """Log error with structured information"""
        if context is None:
            context = {}

        error_category = self.classify_error(error)
        if severity is None:
            severity = self.get_error_severity(error, context)

        # Build structured log entry
        log_data = {
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'error_category': error_category.value,
            'severity': severity.value,
            'timestamp': time.time(),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(),
            **context
        }

        # Log at appropriate level
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error in {operation}: {error}", extra=log_data)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity error in {operation}: {error}", extra=log_data)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium severity error in {operation}: {error}", extra=log_data)
        else:
            self.logger.info(f"Low severity error in {operation}: {error}", extra=log_data)

    def handle_error(self, error: Exception, context: Dict[str, Any] = None,
                    operation: str = "unknown", should_raise: bool = True):
        """Handle error with logging and optional recovery"""
        self.log_error(error, context, operation)

        error_category = self.classify_error(error)
        recovery_strategy = self.get_recovery_strategy(error_category)

        if recovery_strategy:
            try:
                self.logger.info(f"Attempting recovery for {error_category.value} error")
                recovery_strategy()
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}")

        if should_raise:
            raise error

# Global error handler instance
error_handler = ErrorHandler()

def performance_monitor(func):
    """Decorator to monitor function performance and errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            result = func(*args, **kwargs)

            # Log successful execution
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            execution_time = (end_time - start_time) * 1000
            memory_delta = (end_memory - start_memory) / 1024 / 1024

            if execution_time > 1000:  # Log slow operations (>1s)
                error_handler.logger.warning(
                    f"Slow operation: {func.__name__}",
                    extra={
                        'function': func.__name__,
                        'execution_time_ms': execution_time,
                        'memory_delta_mb': memory_delta,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }
                )

            return result

        except Exception as e:
            # Log error with performance context
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000

            error_handler.handle_error(
                e,
                context={
                    'function': func.__name__,
                    'execution_time_ms': execution_time,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                },
                operation=f"{func.__name__}_execution",
                should_raise=True
            )

    return wrapper

@contextmanager
def operation_context(operation_name: str, **metadata):
    """Context manager for consistent operation logging"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    error_handler.logger.info(f"Starting {operation_name}", extra={
        'operation': operation_name,
        'start_time': start_time,
        **metadata
    })

    try:
        yield

        # Log successful completion
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss

        execution_time = (end_time - start_time) * 1000
        memory_delta = (end_memory - start_memory) / 1024 / 1024

        error_handler.logger.info(f"Completed {operation_name}", extra={
            'operation': operation_name,
            'execution_time_ms': execution_time,
            'memory_delta_mb': memory_delta,
            'success': True,
            **metadata
        })

    except Exception as e:
        # Log failure
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000

        error_handler.handle_error(
            e,
            context={
                'operation': operation_name,
                'execution_time_ms': execution_time,
                'success': False,
                **metadata
            },
            operation=operation_name,
            should_raise=True
        )

def safe_execute(func: Callable, *args, fallback=None, **kwargs):
    """Execute function safely with fallback"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler.handle_error(
            e,
            context={'fallback_used': fallback is not None},
            operation=f"safe_execute_{func.__name__}",
            should_raise=False
        )
        return fallback
