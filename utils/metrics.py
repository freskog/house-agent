"""
Performance metrics and timing framework for house-agent.

Provides decorators and utilities for measuring and logging performance metrics
including TTFT, transcription time, TTS time, tool execution time, etc.
"""

import time
import functools
import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum
import threading
from contextlib import asynccontextmanager, contextmanager

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics to track"""
    DURATION = "duration"
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

class OperationType(Enum):
    """Standard operation types for consistent naming"""
    # Voice Processing Pipeline
    VAD_DETECTION = "vad_detection"
    TRANSCRIPTION = "transcription"
    TTS_GENERATION = "tts_generation"
    TTS_STREAMING = "tts_streaming"
    AUDIO_PROCESSING = "audio_processing"
    
    # Agent Processing
    AGENT_TOTAL = "agent_total"
    ROUTER_DECISION = "router_decision"
    NODE_PROCESSING = "node_processing"
    LLM_GENERATION = "llm_generation"
    TOOL_EXECUTION = "tool_execution"
    STATE_TRANSITION = "state_transition"
    
    # Network and I/O
    HTTP_REQUEST = "http_request"
    DATABASE_QUERY = "database_query"
    FILE_IO = "file_io"
    
    # System
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    INITIALIZATION = "initialization"

@dataclass
class PerformanceMetric:
    """Individual performance metric data"""
    operation: str
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    thread_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary for logging"""
        return {
            "operation": self.operation,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "thread_id": self.thread_id,
            "metadata": self.metadata
        }

@dataclass
class TimingContext:
    """Context for timing operations"""
    operation: str
    start_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
class MetricsCollector:
    """Thread-safe metrics collector for aggregating performance data"""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.aggregated_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0,
            'recent_times': deque(maxlen=100)  # Keep last 100 for rolling average
        })
        self._lock = threading.Lock()
        
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric"""
        with self._lock:
            self.metrics.append(metric)
            
            # Update aggregated stats for duration metrics
            if metric.metric_type == MetricType.DURATION:
                stats = self.aggregated_stats[metric.operation]
                stats['count'] += 1
                stats['total_time'] += metric.value
                stats['min_time'] = min(stats['min_time'], metric.value)
                stats['max_time'] = max(stats['max_time'], metric.value)
                stats['avg_time'] = stats['total_time'] / stats['count']
                stats['recent_times'].append(metric.value)
    
    def record_duration(self, operation: str, duration_ms: float, 
                       metadata: Optional[Dict[str, Any]] = None,
                       thread_id: Optional[str] = None) -> None:
        """Record a duration metric"""
        metric = PerformanceMetric(
            operation=operation,
            metric_type=MetricType.DURATION,
            value=duration_ms,
            thread_id=thread_id,
            metadata=metadata or {}
        )
        self.record_metric(metric)
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated statistics"""
        with self._lock:
            if operation:
                return dict(self.aggregated_stats.get(operation, {}))
            return {op: dict(stats) for op, stats in self.aggregated_stats.items()}
    
    def get_recent_metrics(self, operation: Optional[str] = None, 
                          limit: int = 100) -> List[PerformanceMetric]:
        """Get recent metrics"""
        with self._lock:
            if operation:
                return [m for m in list(self.metrics)[-limit:] if m.operation == operation]
            return list(self.metrics)[-limit:]
    
    def clear(self) -> None:
        """Clear all metrics"""
        with self._lock:
            self.metrics.clear()
            self.aggregated_stats.clear()
    
    def get_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        stats = self.get_stats()
        if not stats:
            return "No metrics collected"
        
        lines = ["Performance Summary:"]
        lines.append("=" * 50)
        
        for operation, data in sorted(stats.items()):
            if data['count'] > 0:
                recent_avg = sum(data['recent_times']) / len(data['recent_times']) if data['recent_times'] else 0
                lines.append(f"{operation}:")
                lines.append(f"  Count: {data['count']}")
                lines.append(f"  Avg: {data['avg_time']:.2f}ms")
                lines.append(f"  Recent Avg: {recent_avg:.2f}ms")
                lines.append(f"  Min: {data['min_time']:.2f}ms")
                lines.append(f"  Max: {data['max_time']:.2f}ms")
                lines.append("")
        
        return "\n".join(lines)

# Global metrics collector instance
_global_collector = MetricsCollector()

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector"""
    return _global_collector

def timing_decorator(operation: Optional[str] = None, 
                    include_args: bool = False,
                    metadata_func: Optional[Callable] = None):
    """
    Decorator to automatically time function/method execution.
    
    Args:
        operation: Operation name (defaults to function name)
        include_args: Whether to include function arguments in metadata
        metadata_func: Function to extract custom metadata from args/kwargs
    """
    def decorator(func):
        op_name = operation or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                metadata = {}
                
                if include_args:
                    metadata['args'] = str(args)
                    metadata['kwargs'] = str(kwargs)
                
                if metadata_func:
                    try:
                        metadata.update(metadata_func(*args, **kwargs))
                    except Exception as e:
                        metadata['metadata_error'] = str(e)
                
                try:
                    result = await func(*args, **kwargs)
                    metadata['success'] = True
                    return result
                except Exception as e:
                    metadata['success'] = False
                    metadata['error'] = str(e)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    _global_collector.record_duration(op_name, duration_ms, metadata)
            
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                metadata = {}
                
                if include_args:
                    metadata['args'] = str(args)
                    metadata['kwargs'] = str(kwargs)
                
                if metadata_func:
                    try:
                        metadata.update(metadata_func(*args, **kwargs))
                    except Exception as e:
                        metadata['metadata_error'] = str(e)
                
                try:
                    result = func(*args, **kwargs)
                    metadata['success'] = True
                    return result
                except Exception as e:
                    metadata['success'] = False
                    metadata['error'] = str(e)
                    raise
                finally:
                    duration_ms = (time.time() - start_time) * 1000
                    _global_collector.record_duration(op_name, duration_ms, metadata)
            
            return sync_wrapper
    
    return decorator

@contextmanager
def time_operation(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        _global_collector.record_duration(operation, duration_ms, metadata)

@asynccontextmanager
async def async_time_operation(operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Async context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        _global_collector.record_duration(operation, duration_ms, metadata)

def log_performance(operation: str, duration_ms: float, 
                   metadata: Optional[Dict[str, Any]] = None,
                   log_level: int = logging.INFO) -> None:
    """
    Log performance metrics with structured data.
    
    Args:
        operation: Operation name
        duration_ms: Duration in milliseconds
        metadata: Additional metadata
        log_level: Logging level
    """
    # Record in collector
    _global_collector.record_duration(operation, duration_ms, metadata)
    
    # Create log entry with performance metadata
    extra_data = {
        'operation': operation,
        'duration_ms': duration_ms
    }
    if metadata:
        extra_data.update(metadata)
    
    # Format message based on operation type and duration
    if duration_ms < 100:
        level_indicator = "⚡"  # Fast
    elif duration_ms < 1000:
        level_indicator = "⏱️"   # Normal
    else:
        level_indicator = "🐌"  # Slow
    
    message = f"{level_indicator} {operation} completed in {duration_ms:.2f}ms"
    
    logger.log(log_level, message, extra=extra_data)

def log_ttft(duration_ms: float, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Log Time To First Token (TTFT) metric"""
    log_performance(
        OperationType.TTS_GENERATION.value + "_ttft", 
        duration_ms, 
        metadata, 
        logging.INFO
    )

def log_transcription_performance(audio_duration_ms: float, processing_duration_ms: float,
                                confidence: float = 1.0, text_length: int = 0) -> None:
    """Log transcription performance with real-time factor"""
    real_time_factor = processing_duration_ms / audio_duration_ms if audio_duration_ms > 0 else 0
    
    metadata = {
        'audio_duration_ms': audio_duration_ms,
        'real_time_factor': real_time_factor,
        'confidence': confidence,
        'text_length': text_length
    }
    
    log_performance(OperationType.TRANSCRIPTION.value, processing_duration_ms, metadata)

def log_tts_performance(text_length: int, generation_duration_ms: float, 
                       audio_size_bytes: int = 0, streaming: bool = False) -> None:
    """Log TTS performance metrics"""
    chars_per_second = (text_length / generation_duration_ms * 1000) if generation_duration_ms > 0 else 0
    
    metadata = {
        'text_length': text_length,
        'chars_per_second': chars_per_second,
        'audio_size_bytes': audio_size_bytes,
        'streaming': streaming
    }
    
    operation = OperationType.TTS_STREAMING.value if streaming else OperationType.TTS_GENERATION.value
    log_performance(operation, generation_duration_ms, metadata)

def log_tool_execution(tool_name: str, duration_ms: float, success: bool = True,
                      error: Optional[str] = None) -> None:
    """Log tool execution performance"""
    metadata = {
        'tool_name': tool_name,
        'success': success
    }
    if error:
        metadata['error'] = error
    
    log_performance(OperationType.TOOL_EXECUTION.value, duration_ms, metadata)

def log_agent_performance(total_duration_ms: float, node_durations: Optional[Dict[str, float]] = None,
                         message_count: int = 0, final_domain: Optional[str] = None) -> None:
    """Log overall agent performance"""
    metadata = {
        'message_count': message_count,
        'final_domain': final_domain
    }
    if node_durations:
        metadata['node_durations'] = node_durations
    
    log_performance(OperationType.AGENT_TOTAL.value, total_duration_ms, metadata)

# Convenience functions for common timing patterns
def start_timer() -> float:
    """Start a timer and return the start time"""
    return time.time()

def end_timer(start_time: float) -> float:
    """End a timer and return duration in milliseconds"""
    return (time.time() - start_time) * 1000

class TimerContext:
    """Reusable timer context for complex timing scenarios"""
    
    def __init__(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time: Optional[float] = None
        self.checkpoints: List[tuple] = []
    
    def start(self) -> 'TimerContext':
        """Start the timer"""
        self.start_time = time.time()
        return self
    
    def checkpoint(self, name: str) -> 'TimerContext':
        """Add a checkpoint"""
        if self.start_time is not None:
            elapsed = (time.time() - self.start_time) * 1000
            self.checkpoints.append((name, elapsed))
        return self
    
    def end(self) -> float:
        """End timing and log the result"""
        if self.start_time is None:
            return 0.0
        
        duration_ms = (time.time() - self.start_time) * 1000
        
        # Add checkpoints to metadata
        if self.checkpoints:
            self.metadata['checkpoints'] = dict(self.checkpoints)
        
        log_performance(self.operation, duration_ms, self.metadata)
        return duration_ms 