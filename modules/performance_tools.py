"""
Performance monitoring and optimization tools for SkyPin
Provides performance monitoring, profiling, and optimization utilities
"""

import time
import psutil
import threading
from typing import Dict, List, Optional, Callable, Any
import logging
from datetime import datetime, timezone
import numpy as np
from contextlib import contextmanager
import functools
import gc
import tracemalloc
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors performance metrics and provides optimization tools."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {}
        self.start_time = time.time()
        self.memory_usage = []
        self.cpu_usage = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 1.0):
        """
        Start performance monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        try:
            if self.monitoring:
                return
            
            self.monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, 
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
            
            logger.info("Performance monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        try:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5.0)
            
            logger.info("Performance monitoring stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop monitoring: {e}")
    
    def _monitor_loop(self, interval: float):
        """Monitoring loop."""
        try:
            while self.monitoring:
                # Collect metrics
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                
                # Store metrics
                timestamp = time.time()
                self.memory_usage.append({
                    'timestamp': timestamp,
                    'used': memory_info.used,
                    'available': memory_info.available,
                    'percent': memory_info.percent
                })
                
                self.cpu_usage.append({
                    'timestamp': timestamp,
                    'percent': cpu_percent
                })
                
                # Keep only recent data (last 1000 points)
                if len(self.memory_usage) > 1000:
                    self.memory_usage = self.memory_usage[-1000:]
                if len(self.cpu_usage) > 1000:
                    self.cpu_usage = self.cpu_usage[-1000:]
                
                time.sleep(interval)
                
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
    
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics."""
        try:
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            disk_usage = psutil.disk_usage('/')
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'memory': {
                    'used': memory_info.used,
                    'available': memory_info.available,
                    'percent': memory_info.percent,
                    'total': memory_info.total
                },
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count()
                },
                'disk': {
                    'used': disk_usage.used,
                    'free': disk_usage.free,
                    'total': disk_usage.total,
                    'percent': (disk_usage.used / disk_usage.total) * 100
                },
                'uptime': time.time() - self.start_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {}
    
    def get_historical_metrics(self) -> Dict:
        """Get historical performance metrics."""
        try:
            return {
                'memory_usage': self.memory_usage,
                'cpu_usage': self.cpu_usage,
                'monitoring_duration': time.time() - self.start_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get historical metrics: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        try:
            current_metrics = self.get_current_metrics()
            historical_metrics = self.get_historical_metrics()
            
            summary = {
                'current': current_metrics,
                'historical': historical_metrics,
                'summary': {
                    'avg_memory_usage': 0.0,
                    'max_memory_usage': 0.0,
                    'avg_cpu_usage': 0.0,
                    'max_cpu_usage': 0.0,
                    'monitoring_duration': 0.0
                }
            }
            
            # Calculate averages and maximums
            if self.memory_usage:
                memory_percents = [m['percent'] for m in self.memory_usage]
                summary['summary']['avg_memory_usage'] = np.mean(memory_percents)
                summary['summary']['max_memory_usage'] = np.max(memory_percents)
            
            if self.cpu_usage:
                cpu_percents = [c['percent'] for c in self.cpu_usage]
                summary['summary']['avg_cpu_usage'] = np.mean(cpu_percents)
                summary['summary']['max_cpu_usage'] = np.max(cpu_percents)
            
            summary['summary']['monitoring_duration'] = time.time() - self.start_time
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}

class PerformanceProfiler:
    """Profiles function execution time and memory usage."""
    
    def __init__(self):
        """Initialize profiler."""
        self.profiles = {}
        
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                # Store profile data
                if func.__name__ not in self.profiles:
                    self.profiles[func.__name__] = []
                
                self.profiles[func.__name__].append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'execution_time': execution_time,
                    'memory_delta': memory_delta,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                })
                
                logger.debug(f"Profiled {func.__name__}: {execution_time:.4f}s, {memory_delta} bytes")
        
        return wrapper
    
    def get_function_profiles(self) -> Dict:
        """Get function profiles."""
        try:
            profiles_summary = {}
            
            for func_name, profile_data in self.profiles.items():
                if profile_data:
                    execution_times = [p['execution_time'] for p in profile_data]
                    memory_deltas = [p['memory_delta'] for p in profile_data]
                    
                    profiles_summary[func_name] = {
                        'call_count': len(profile_data),
                        'avg_execution_time': np.mean(execution_times),
                        'min_execution_time': np.min(execution_times),
                        'max_execution_time': np.max(execution_times),
                        'total_execution_time': np.sum(execution_times),
                        'avg_memory_delta': np.mean(memory_deltas),
                        'max_memory_delta': np.max(memory_deltas),
                        'recent_calls': profile_data[-10:]  # Last 10 calls
                    }
            
            return profiles_summary
            
        except Exception as e:
            logger.error(f"Failed to get function profiles: {e}")
            return {}
    
    def clear_profiles(self):
        """Clear all profiles."""
        self.profiles.clear()
        logger.info("Profiles cleared")

class MemoryProfiler:
    """Profiles memory usage with detailed tracking."""
    
    def __init__(self):
        """Initialize memory profiler."""
        self.snapshots = []
        self.tracemalloc_started = False
        
    def start_tracemalloc(self):
        """Start tracemalloc for detailed memory tracking."""
        try:
            if not self.tracemalloc_started:
                tracemalloc.start()
                self.tracemalloc_started = True
                logger.info("Tracemalloc started")
        except Exception as e:
            logger.error(f"Failed to start tracemalloc: {e}")
    
    def stop_tracemalloc(self):
        """Stop tracemalloc."""
        try:
            if self.tracemalloc_started:
                tracemalloc.stop()
                self.tracemalloc_started = False
                logger.info("Tracemalloc stopped")
        except Exception as e:
            logger.error(f"Failed to stop tracemalloc: {e}")
    
    def take_snapshot(self, label: str = None):
        """Take a memory snapshot."""
        try:
            if self.tracemalloc_started:
                snapshot = tracemalloc.take_snapshot()
                self.snapshots.append({
                    'label': label or f"snapshot_{len(self.snapshots)}",
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'snapshot': snapshot
                })
                logger.debug(f"Memory snapshot taken: {label}")
        except Exception as e:
            logger.error(f"Failed to take snapshot: {e}")
    
    def get_memory_stats(self) -> Dict:
        """Get memory statistics."""
        try:
            if not self.tracemalloc_started:
                return {'error': 'Tracemalloc not started'}
            
            current_snapshot = tracemalloc.take_snapshot()
            
            # Get top memory allocations
            top_stats = current_snapshot.statistics('lineno')
            
            stats = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_memory': sum(stat.size for stat in top_stats),
                'total_blocks': sum(stat.count for stat in top_stats),
                'top_allocations': []
            }
            
            # Get top 10 allocations
            for stat in top_stats[:10]:
                stats['top_allocations'].append({
                    'filename': stat.traceback.format()[0],
                    'size': stat.size,
                    'count': stat.count,
                    'size_per_block': stat.size / stat.count
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {'error': str(e)}
    
    def compare_snapshots(self, snapshot1_label: str, snapshot2_label: str) -> Dict:
        """Compare two memory snapshots."""
        try:
            snapshot1 = None
            snapshot2 = None
            
            for snapshot in self.snapshots:
                if snapshot['label'] == snapshot1_label:
                    snapshot1 = snapshot['snapshot']
                elif snapshot['label'] == snapshot2_label:
                    snapshot2 = snapshot['snapshot']
            
            if not snapshot1 or not snapshot2:
                return {'error': 'Snapshots not found'}
            
            # Compare snapshots
            comparison = snapshot2.compare_to(snapshot1, 'lineno')
            
            comparison_data = {
                'snapshot1': snapshot1_label,
                'snapshot2': snapshot2_label,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'differences': []
            }
            
            for stat in comparison[:10]:  # Top 10 differences
                comparison_data['differences'].append({
                    'filename': stat.traceback.format()[0],
                    'size_diff': stat.size_diff,
                    'count_diff': stat.count_diff,
                    'size_per_block_diff': stat.size_diff / stat.count_diff if stat.count_diff != 0 else 0
                })
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Failed to compare snapshots: {e}")
            return {'error': str(e)}

class OptimizationTools:
    """Provides optimization tools and recommendations."""
    
    def __init__(self):
        """Initialize optimization tools."""
        self.optimization_history = []
        
    def analyze_performance_bottlenecks(self, profiles: Dict) -> Dict:
        """Analyze performance bottlenecks."""
        try:
            analysis = {
                'bottlenecks': [],
                'recommendations': [],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Find slowest functions
            slow_functions = []
            for func_name, profile_data in profiles.items():
                if profile_data.get('call_count', 0) > 0:
                    avg_time = profile_data.get('avg_execution_time', 0)
                    total_time = profile_data.get('total_execution_time', 0)
                    
                    if avg_time > 1.0:  # Functions taking more than 1 second
                        slow_functions.append({
                            'function': func_name,
                            'avg_time': avg_time,
                            'total_time': total_time,
                            'call_count': profile_data.get('call_count', 0)
                        })
            
            # Sort by total time
            slow_functions.sort(key=lambda x: x['total_time'], reverse=True)
            analysis['bottlenecks'] = slow_functions
            
            # Generate recommendations
            for bottleneck in slow_functions[:5]:  # Top 5 bottlenecks
                if bottleneck['avg_time'] > 5.0:
                    analysis['recommendations'].append(
                        f"Function '{bottleneck['function']}' is very slow ({bottleneck['avg_time']:.2f}s avg). "
                        "Consider optimizing or caching results."
                    )
                elif bottleneck['call_count'] > 100:
                    analysis['recommendations'].append(
                        f"Function '{bottleneck['function']}' is called frequently ({bottleneck['call_count']} times). "
                        "Consider memoization or reducing call frequency."
                    )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Bottleneck analysis failed: {e}")
            return {'error': str(e)}
    
    def optimize_memory_usage(self, memory_stats: Dict) -> Dict:
        """Analyze memory usage and provide optimization recommendations."""
        try:
            optimization = {
                'memory_issues': [],
                'recommendations': [],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Check for memory issues
            if memory_stats.get('total_memory', 0) > 100 * 1024 * 1024:  # 100MB
                optimization['memory_issues'].append("High memory usage detected")
                optimization['recommendations'].append("Consider using generators or streaming for large datasets")
            
            # Check for large allocations
            for allocation in memory_stats.get('top_allocations', []):
                if allocation['size'] > 10 * 1024 * 1024:  # 10MB
                    optimization['memory_issues'].append(f"Large allocation in {allocation['filename']}")
                    optimization['recommendations'].append(
                        f"Consider optimizing memory usage in {allocation['filename']}"
                    )
            
            return optimization
            
        except Exception as e:
            logger.error(f"Memory optimization analysis failed: {e}")
            return {'error': str(e)}
    
    def get_optimization_recommendations(self, performance_data: Dict) -> List[str]:
        """Get optimization recommendations based on performance data."""
        try:
            recommendations = []
            
            # Check CPU usage
            if performance_data.get('cpu', {}).get('percent', 0) > 80:
                recommendations.append("High CPU usage detected. Consider using multiprocessing or optimizing algorithms.")
            
            # Check memory usage
            if performance_data.get('memory', {}).get('percent', 0) > 80:
                recommendations.append("High memory usage detected. Consider memory optimization or garbage collection.")
            
            # Check disk usage
            if performance_data.get('disk', {}).get('percent', 0) > 90:
                recommendations.append("High disk usage detected. Consider cleaning up temporary files.")
            
            # Check for long-running processes
            uptime = performance_data.get('uptime', 0)
            if uptime > 3600:  # 1 hour
                recommendations.append("Long-running process detected. Consider restarting to free up resources.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return [f"Error generating recommendations: {e}"]

@contextmanager
def performance_context(label: str = None):
    """Context manager for performance monitoring."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        execution_time = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logger.info(f"Performance context '{label}': {execution_time:.4f}s, {memory_delta} bytes")

def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution."""
    profiler = PerformanceProfiler()
    return profiler.profile_function(func)

def optimize_image_processing(image: np.ndarray, target_size: tuple = None) -> np.ndarray:
    """Optimize image for processing."""
    try:
        # Resize if too large
        if target_size and image.shape[:2] != target_size:
            from modules.image_enhancer import resize_image
            image = resize_image(image, target_size)
        
        # Convert to appropriate data type
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        return image
        
    except Exception as e:
        logger.error(f"Image optimization failed: {e}")
        return image

def clear_memory_cache():
    """Clear memory cache and run garbage collection."""
    try:
        gc.collect()
        logger.info("Memory cache cleared")
    except Exception as e:
        logger.error(f"Failed to clear memory cache: {e}")

# Global instances
_performance_monitor = None
_performance_profiler = None
_memory_profiler = None
_optimization_tools = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

def get_performance_profiler() -> PerformanceProfiler:
    """Get performance profiler instance."""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler

def get_memory_profiler() -> MemoryProfiler:
    """Get memory profiler instance."""
    global _memory_profiler
    if _memory_profiler is None:
        _memory_profiler = MemoryProfiler()
    return _memory_profiler

def get_optimization_tools() -> OptimizationTools:
    """Get optimization tools instance."""
    global _optimization_tools
    if _optimization_tools is None:
        _optimization_tools = OptimizationTools()
    return _optimization_tools

def start_performance_monitoring(interval: float = 1.0):
    """Start performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring(interval)

def stop_performance_monitoring():
    """Stop performance monitoring."""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()

def get_performance_summary() -> Dict:
    """Get performance summary."""
    monitor = get_performance_monitor()
    return monitor.get_performance_summary()

def get_function_profiles() -> Dict:
    """Get function profiles."""
    profiler = get_performance_profiler()
    return profiler.get_function_profiles()

def analyze_performance_bottlenecks() -> Dict:
    """Analyze performance bottlenecks."""
    profiler = get_performance_profiler()
    optimization_tools = get_optimization_tools()
    profiles = profiler.get_function_profiles()
    return optimization_tools.analyze_performance_bottlenecks(profiles)