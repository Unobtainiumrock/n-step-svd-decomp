"""
Performance monitoring and optimization utilities.
"""

import time
import psutil
import numpy as np
from typing import Dict, Any, Optional, Callable
import structlog

logger = structlog.get_logger(__name__)


class PerformanceMonitor:
    """
    Monitor and track system performance metrics.
    """
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics_history: Dict[str, list] = {}
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation_name: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation_name: Name of the operation to time
        """
        self.start_times[operation_name] = time.time()
        logger.debug("Started timing operation", operation=operation_name)
    
    def end_timer(self, operation_name: str) -> float:
        """
        End timing an operation and return duration.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Duration in seconds
        """
        if operation_name not in self.start_times:
            logger.warning("No start time found for operation", operation=operation_name)
            return 0.0
        
        duration = time.time() - self.start_times[operation_name]
        
        # Store in history
        if operation_name not in self.metrics_history:
            self.metrics_history[operation_name] = []
        self.metrics_history[operation_name].append(duration)
        
        logger.info("Operation completed", operation=operation_name, duration_ms=duration * 1000)
        return duration
    
    def time_operation(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Time a function execution.
        
        Args:
            operation_name: Name of the operation
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        self.start_timer(operation_name)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            self.end_timer(operation_name)
    
    def get_system_metrics(self) -> Dict[str, float]:
        """
        Get current system performance metrics.
        
        Returns:
            Dictionary of system metrics
        """
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage_mb = memory.used / (1024 * 1024)
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_gb = disk.used / (1024 * 1024 * 1024)
        disk_percent = (disk.used / disk.total) * 100
        
        metrics = {
            'cpu_usage_percent': cpu_percent,
            'memory_usage_mb': memory_usage_mb,
            'memory_usage_percent': memory_percent,
            'disk_usage_gb': disk_usage_gb,
            'disk_usage_percent': disk_percent,
        }
        
        logger.debug("System metrics collected", metrics=metrics)
        return metrics
    
    def get_operation_statistics(self, operation_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary of operation statistics
        """
        if operation_name not in self.metrics_history:
            return {}
        
        durations = self.metrics_history[operation_name]
        
        stats = {
            'count': len(durations),
            'mean_ms': np.mean(durations) * 1000,
            'std_ms': np.std(durations) * 1000,
            'min_ms': np.min(durations) * 1000,
            'max_ms': np.max(durations) * 1000,
            'median_ms': np.median(durations) * 1000,
            'p95_ms': np.percentile(durations, 95) * 1000,
            'p99_ms': np.percentile(durations, 99) * 1000,
        }
        
        logger.info("Operation statistics calculated", operation=operation_name, stats=stats)
        return stats
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all operations.
        
        Returns:
            Dictionary mapping operation names to their statistics
        """
        all_stats = {}
        for operation_name in self.metrics_history.keys():
            all_stats[operation_name] = self.get_operation_statistics(operation_name)
        
        return all_stats
    
    def clear_history(self, operation_name: Optional[str] = None) -> None:
        """
        Clear performance history.
        
        Args:
            operation_name: Specific operation to clear, or None for all
        """
        if operation_name is None:
            self.metrics_history.clear()
            self.start_times.clear()
            logger.info("Cleared all performance history")
        else:
            if operation_name in self.metrics_history:
                del self.metrics_history[operation_name]
            if operation_name in self.start_times:
                del self.start_times[operation_name]
            logger.info("Cleared performance history for operation", operation=operation_name)
    
    def benchmark_embedding_operations(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Benchmark embedding-related operations.
        
        Args:
            embeddings: Embeddings to benchmark
            
        Returns:
            Dictionary of benchmark results
        """
        logger.info("Benchmarking embedding operations", shape=embeddings.shape)
        
        benchmarks = {}
        
        # Benchmark normalization
        def normalize():
            return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        benchmarks['normalize_ms'] = self.time_operation('normalize', normalize) * 1000
        
        # Benchmark dot product
        def dot_product():
            return np.dot(embeddings, embeddings.T)
        
        benchmarks['dot_product_ms'] = self.time_operation('dot_product', dot_product) * 1000
        
        # Benchmark matrix multiplication
        def matrix_mult():
            return embeddings @ embeddings.T
        
        benchmarks['matrix_mult_ms'] = self.time_operation('matrix_mult', matrix_mult) * 1000
        
        logger.info("Embedding operations benchmarked", benchmarks=benchmarks)
        return benchmarks
