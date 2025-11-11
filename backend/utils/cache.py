"""
Advanced multi-level caching system for Frankenstino AI
Implements LRU, LFU, and adaptive caching strategies with memory management
"""

import os
import time
import threading
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import OrderedDict, defaultdict
import logging
import psutil
import gc

logger = logging.getLogger(__name__)

class CacheEntry:
    """Represents a cache entry with metadata."""

    def __init__(self, key: str, value: Any, size_bytes: int = 0,
                 ttl_seconds: Optional[int] = None, access_count: int = 0):
        self.key = key
        self.value = value
        self.size_bytes = size_bytes
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = access_count
        self.ttl_seconds = ttl_seconds
        self.expires_at = self.created_at + ttl_seconds if ttl_seconds else None

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return self.expires_at is not None and time.time() > self.expires_at

    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at

    def idle_seconds(self) -> float:
        """Get idle time (time since last access) in seconds."""
        return time.time() - self.last_accessed

class SmartCache:
    """
    Advanced multi-level cache with LRU, LFU, and adaptive eviction policies.
    Supports memory limits, TTL, persistence, and performance monitoring.
    """

    def __init__(self, max_memory_mb: int = 500, cache_dir: str = "data/cache",
                 policy: str = "adaptive", enable_persistence: bool = True,
                 cleanup_interval_seconds: int = 300):
        """
        Initialize smart cache.

        Args:
            max_memory_mb: Maximum memory usage in MB
            cache_dir: Directory for persistent cache storage
            policy: Eviction policy ('lru', 'lfu', 'adaptive')
            enable_persistence: Whether to persist cache to disk
            cleanup_interval_seconds: How often to run cleanup (0 to disable)
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy
        self.enable_persistence = enable_persistence

        # Core storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.memory_usage = 0

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.sets = 0

        # Multi-level cache layers
        self.l1_cache: OrderedDict[str, Any] = OrderedDict()  # Fast access
        self.l2_cache: Dict[str, CacheEntry] = {}  # Full entries

        # Adaptive policy tracking
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.policy_weights = {'lru': 1.0, 'lfu': 1.0}

        # Threading
        self.lock = threading.RLock()
        self.cleanup_thread = None

        # Start cleanup thread if enabled
        if cleanup_interval_seconds > 0:
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                args=(cleanup_interval_seconds,),
                daemon=True
            )
            self.cleanup_thread.start()

        logger.info(f"Initialized SmartCache: {max_memory_mb}MB limit, {policy} policy")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        with self.lock:
            # Check L1 cache first (fast path)
            if key in self.l1_cache:
                self.hits += 1
                entry = self.l2_cache[key]
                entry.touch()
                self._update_access_pattern(key)
                return self.l1_cache[key]

            # Check L2 cache
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                if not entry.is_expired():
                    self.hits += 1
                    entry.touch()
                    # Promote to L1
                    self.l1_cache[key] = entry.value
                    self._update_access_pattern(key)
                    return entry.value
                else:
                    # Remove expired entry
                    self._remove_entry(key)

            self.misses += 1
            return default

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
            size_bytes: Optional[int] = None) -> bool:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to store
            ttl_seconds: Time-to-live in seconds
            size_bytes: Size of value in bytes (estimated if not provided)

        Returns:
            True if stored successfully
        """
        with self.lock:
            # Estimate size if not provided
            if size_bytes is None:
                size_bytes = self._estimate_size(value)

            # Check if we need to evict
            if self.memory_usage + size_bytes > self.max_memory_bytes:
                self._evict_to_fit(size_bytes)

            # Create entry
            entry = CacheEntry(key, value, size_bytes, ttl_seconds)

            # Store in L2
            if key in self.l2_cache:
                old_entry = self.l2_cache[key]
                self.memory_usage -= old_entry.size_bytes
            else:
                self.sets += 1

            self.l2_cache[key] = entry
            self.memory_usage += size_bytes

            # Store in L1 (limited size)
            if len(self.l1_cache) < 100:  # Keep L1 small for speed
                self.l1_cache[key] = value
            else:
                # Remove oldest from L1
                oldest_key, _ = self.l1_cache.popitem(last=False)
                if oldest_key != key:
                    del self.l1_cache[oldest_key]

            return True

    def delete(self, key: str) -> bool:
        """
        Remove key from cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was removed
        """
        with self.lock:
            return self._remove_entry(key)

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.memory_usage = 0
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.sets = 0
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                'memory_usage_mb': self.memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_usage_percent': (self.memory_usage / self.max_memory_bytes) * 100,
                'total_entries': len(self.l2_cache),
                'l1_entries': len(self.l1_cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'sets': self.sets,
                'policy': self.policy
            }

    def optimize_policy(self):
        """Adaptively optimize cache policy based on access patterns."""
        if self.policy != 'adaptive':
            return

        # Analyze access patterns
        lru_score = self._calculate_lru_effectiveness()
        lfu_score = self._calculate_lfu_effectiveness()

        # Adjust weights based on performance
        total_score = lru_score + lfu_score
        if total_score > 0:
            self.policy_weights['lru'] = lru_score / total_score
            self.policy_weights['lfu'] = lfu_score / total_score

        logger.info(f"Optimized cache policy weights: LRU={self.policy_weights['lru']:.2f}, LFU={self.policy_weights['lfu']:.2f}")

    def save_to_disk(self, filename: str = "cache_snapshot.pkl"):
        """Persist cache to disk."""
        if not self.enable_persistence:
            return

        filepath = self.cache_dir / filename
        with self.lock:
            data = {
                'l2_cache': self.l2_cache,
                'memory_usage': self.memory_usage,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'sets': self.sets,
                'policy_weights': self.policy_weights
            }

            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Cache saved to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

    def load_from_disk(self, filename: str = "cache_snapshot.pkl"):
        """Load cache from disk."""
        if not self.enable_persistence:
            return

        filepath = self.cache_dir / filename
        if not filepath.exists():
            return

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            with self.lock:
                self.l2_cache = data.get('l2_cache', {})
                self.memory_usage = data.get('memory_usage', 0)
                self.hits = data.get('hits', 0)
                self.misses = data.get('misses', 0)
                self.evictions = data.get('evictions', 0)
                self.sets = data.get('sets', 0)
                self.policy_weights = data.get('policy_weights', {'lru': 1.0, 'lfu': 1.0})

                # Rebuild L1 cache
                self.l1_cache.clear()
                for key, entry in list(self.l2_cache.items())[:100]:  # Load top 100 to L1
                    if not entry.is_expired():
                        self.l1_cache[key] = entry.value

            logger.info(f"Cache loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")

    def _evict_to_fit(self, required_bytes: int):
        """Evict entries to make room for new data."""
        bytes_needed = required_bytes - (self.max_memory_bytes - self.memory_usage)
        if bytes_needed <= 0:
            return

        evicted_bytes = 0
        eviction_candidates = []

        # Build candidate list based on policy
        for key, entry in self.l2_cache.items():
            if entry.is_expired():
                # Always evict expired entries first
                eviction_candidates.append((key, entry, 0))  # Priority 0 (highest)
            else:
                priority = self._calculate_eviction_priority(entry)
                eviction_candidates.append((key, entry, priority))

        # Sort by priority (lower number = higher priority for eviction)
        eviction_candidates.sort(key=lambda x: x[2])

        # Evict until we have enough space
        for key, entry, _ in eviction_candidates:
            if evicted_bytes >= bytes_needed:
                break

            self._remove_entry(key)
            evicted_bytes += entry.size_bytes
            self.evictions += 1

        logger.debug(f"Evicted {len(eviction_candidates)} entries, freed {evicted_bytes} bytes")

    def _calculate_eviction_priority(self, entry: CacheEntry) -> float:
        """Calculate eviction priority based on current policy."""
        if self.policy == 'lru':
            return entry.idle_seconds()  # More idle = higher priority for eviction
        elif self.policy == 'lfu':
            return 1.0 / (entry.access_count + 1)  # Less accessed = higher priority
        elif self.policy == 'adaptive':
            lru_priority = entry.idle_seconds() * self.policy_weights['lru']
            lfu_priority = (1.0 / (entry.access_count + 1)) * self.policy_weights['lfu']
            return lru_priority + lfu_priority
        else:
            return entry.idle_seconds()  # Default to LRU

    def _calculate_lru_effectiveness(self) -> float:
        """Calculate how effective LRU policy has been."""
        if not self.access_patterns:
            return 1.0

        # Simple heuristic: higher scores for more recent access patterns
        recent_accesses = []
        for pattern in self.access_patterns.values():
            if pattern:
                recent_accesses.extend(pattern[-10:])  # Last 10 accesses

        if not recent_accesses:
            return 1.0

        # Score based on recency
        max_time = max(recent_accesses)
        min_time = min(recent_accesses)
        time_range = max_time - min_time

        return 1.0 if time_range == 0 else sum(recent_accesses) / (len(recent_accesses) * max_time)

    def _calculate_lfu_effectiveness(self) -> float:
        """Calculate how effective LFU policy has been."""
        if not self.l2_cache:
            return 1.0

        total_accesses = sum(entry.access_count for entry in self.l2_cache.values())
        avg_accesses = total_accesses / len(self.l2_cache)

        # Higher score for higher average accesses (more frequently used cache)
        return min(avg_accesses / 10.0, 1.0)  # Cap at 1.0

    def _update_access_pattern(self, key: str):
        """Update access pattern tracking."""
        self.access_patterns[key].append(time.time())
        # Keep only last 100 accesses per key
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]

    def _remove_entry(self, key: str) -> bool:
        """Remove entry from all cache levels."""
        if key in self.l1_cache:
            del self.l1_cache[key]

        if key in self.l2_cache:
            entry = self.l2_cache[key]
            self.memory_usage -= entry.size_bytes
            del self.l2_cache[key]
            return True

        return False

    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of an object in bytes."""
        try:
            # Use sys.getsizeof for basic estimation
            import sys
            size = sys.getsizeof(obj)

            # Add size of contents for containers
            if hasattr(obj, '__len__') and hasattr(obj, '__iter__'):
                if isinstance(obj, (list, tuple)):
                    size += sum(self._estimate_size(item) for item in obj)
                elif isinstance(obj, dict):
                    size += sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())

            return size
        except:
            return 1024  # Default estimate

    def _cleanup_worker(self, interval_seconds: int):
        """Background cleanup worker."""
        while True:
            time.sleep(interval_seconds)
            try:
                self._cleanup_expired()
                self.optimize_policy()

                # Periodic persistence
                if self.enable_persistence:
                    self.save_to_disk()

            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def _cleanup_expired(self):
        """Remove expired entries."""
        with self.lock:
            expired_keys = []
            for key, entry in self.l2_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_entry(key)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

class MemoryMonitor:
    """Monitor system memory usage and provide recommendations."""

    def __init__(self, cache: SmartCache):
        self.cache = cache
        self.system_memory_warning = 80  # Percent

    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        cache_stats = self.cache.get_stats()

        # System memory
        system_memory = psutil.virtual_memory()
        system_percent = system_memory.percent

        return {
            'cache_memory_mb': cache_stats['memory_usage_mb'],
            'cache_memory_percent': cache_stats['memory_usage_percent'],
            'system_memory_percent': system_percent,
            'system_memory_available_mb': system_memory.available / (1024 * 1024),
            'memory_pressure': system_percent > self.system_memory_warning,
            'recommendations': self._get_recommendations(system_percent, cache_stats)
        }

    def _get_recommendations(self, system_percent: float, cache_stats: Dict) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []

        if system_percent > self.system_memory_warning:
            recommendations.append("High system memory usage - consider reducing cache size")

        if cache_stats['memory_usage_percent'] > 90:
            recommendations.append("Cache near capacity - consider increasing cache size or enabling eviction")

        if cache_stats['hit_rate'] < 0.5:
            recommendations.append("Low cache hit rate - consider adjusting cache policy or size")

        if len(cache_stats) > 10000:  # Arbitrary large number
            recommendations.append("Large number of cache entries - consider memory optimization")

        return recommendations

# Global cache instance
_cache_instance = None
_monitor_instance = None

def get_cache() -> SmartCache:
    """Get global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SmartCache()
    return _cache_instance

def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        cache = get_cache()
        _monitor_instance = MemoryMonitor(cache)
    return _monitor_instance
