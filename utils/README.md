# House Agent Logging and Metrics Infrastructure

This directory contains the centralized logging and performance metrics infrastructure for the house-agent system.

## Overview

The infrastructure provides:
- **Centralized Logging**: Standardized logging configuration with support for different environments
- **Performance Metrics**: Comprehensive timing and performance measurement tools
- **Configuration Management**: Environment-based configuration with sensible defaults

## Quick Start

```python
from utils import setup_logging, get_logger, timing_decorator, log_performance

# Setup logging (automatically configured based on environment)
logger = get_logger(__name__)

# Use basic logging
logger.info("Application started")
logger.debug("Debug information")
logger.error("Something went wrong")

# Automatic timing with decorator
@timing_decorator(operation="my_function")
def my_function():
    # Your code here
    pass

# Manual performance logging
log_performance("operation_name", duration_ms=123.45, metadata={"key": "value"})
```

## Features

### 1. Centralized Logging Configuration

```python
from utils.logging_config import setup_logging

# Development: colored console output with DEBUG level
setup_logging(level="DEBUG", format_type="standard", enable_colors=True)

# Production: structured JSON logging with INFO level
setup_logging(level="INFO", format_type="structured", log_file="app.log")

# Testing: minimal logging with WARNING level
setup_logging(level="WARNING", format_type="minimal")
```

### 2. Performance Metrics and Timing

```python
from utils.metrics import timing_decorator, time_operation, log_performance

# Automatic timing with decorators
@timing_decorator("database_query")
async def query_database():
    # Function automatically timed
    pass

# Manual timing with context managers
with time_operation("complex_operation", {"type": "batch"}):
    # Your code here
    pass

# Specialized performance logging
from utils.metrics import log_transcription_performance, log_tts_performance

log_transcription_performance(
    audio_duration_ms=2000,
    processing_duration_ms=250,
    confidence=0.95
)
```

### 3. Environment-Based Configuration

The system automatically configures itself based on the `HOUSE_AGENT_ENV` environment variable:

- **development** (default): DEBUG level, colored console, file logging
- **production**: INFO level, structured JSON, file logging
- **testing**: WARNING level, minimal format, no file logging

### 4. Metrics Collection and Analysis

```python
from utils.metrics import get_metrics_collector

collector = get_metrics_collector()

# Get performance statistics
stats = collector.get_stats()
print(collector.get_summary_report())

# Get recent metrics for analysis
recent = collector.get_recent_metrics("transcription", limit=50)
```

## Environment Variables

Configure the system using environment variables:

### Logging Configuration
- `LOG_LEVEL`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `LOG_FORMAT`: Format type (standard, structured, minimal)
- `LOG_FILE`: Log file path (optional)
- `LOG_ENABLE_COLORS`: Enable colored console output (true/false)

### Metrics Configuration
- `METRICS_ENABLED`: Enable metrics collection (true/false)
- `METRICS_LOG_PERFORMANCE`: Enable performance logging (true/false)
- `METRICS_SLOW_THRESHOLD_MS`: Threshold for logging slow operations

### Environment
- `HOUSE_AGENT_ENV`: Environment mode (development, production, testing)

## Usage Examples

### Basic Logging
```python
from utils import get_logger

logger = get_logger(__name__)

logger.info("User connected")
logger.warning("High memory usage detected")
logger.error("Failed to connect to database", extra={"error_code": 500})
```

### Performance Monitoring
```python
from utils.metrics import timing_decorator, log_tts_performance

@timing_decorator("audio_processing")
async def process_audio(audio_data):
    # Processing automatically timed
    return processed_data

# Specialized TTS logging
log_tts_performance(
    text_length=100,
    generation_duration_ms=500,
    audio_size_bytes=24000,
    streaming=True
)
```

### Complex Timing
```python
from utils.metrics import TimerContext

timer = TimerContext("multi_phase_operation", {"user_id": "123"})
timer.start()

# Phase 1
process_phase_1()
timer.checkpoint("phase_1_complete")

# Phase 2
process_phase_2()
timer.checkpoint("phase_2_complete")

# Final timing
total_time = timer.end()  # Automatically logs with checkpoints
```

## Integration with Existing Code

To integrate with existing modules:

1. **Replace print statements** with logger calls:
   ```python
   # Before
   print(f"Processing {item}")
   
   # After
   logger.info(f"Processing {item}")
   ```

2. **Add timing to critical paths**:
   ```python
   # Before
   def expensive_operation():
       # ... code ...
       
   # After
   @timing_decorator("expensive_operation")
   def expensive_operation():
       # ... code ...
   ```

3. **Use specialized logging** for domain-specific metrics:
   ```python
   # For audio processing
   log_transcription_performance(audio_duration, processing_time, confidence)
   
   # For agent operations
   log_tool_execution("spotify_play", duration_ms, success=True)
   ```

## Best Practices

1. **Use appropriate log levels**:
   - `ERROR`: System failures, exceptions
   - `WARNING`: Degraded performance, fallbacks
   - `INFO`: Major state transitions, user actions
   - `DEBUG`: Detailed processing steps

2. **Add context with metadata**:
   ```python
   logger.info("Request processed", extra={"user_id": "123", "duration_ms": 45})
   ```

3. **Time critical operations**:
   - Voice processing pipeline (transcription, TTS)
   - Agent decision making
   - Tool execution
   - Network requests

4. **Use structured logging in production**:
   ```bash
   export HOUSE_AGENT_ENV=production
   export LOG_FORMAT=structured
   ```

## Files

- `logging_config.py`: Centralized logging configuration and formatters
- `metrics.py`: Performance metrics, timing decorators, and collectors
- `config.py`: Configuration management and environment detection
- `example_usage.py`: Examples and usage patterns
- `README.md`: This documentation

## Migration from Print Statements

Phase 1 has established the infrastructure. Next phases will:

1. Replace print statements in `audio_agent.py` and `audio/server/websocket.py`
2. Add performance timing to voice processing pipeline
3. Update all node modules to use proper logging
4. Implement comprehensive performance monitoring

The infrastructure is now ready for seamless integration throughout the codebase! 