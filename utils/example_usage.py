"""
Example usage of the logging and metrics infrastructure.

This script demonstrates how to use the centralized logging and performance
metrics throughout the house-agent codebase.
"""

import asyncio
import time
from utils.logging_config import setup_logging, get_logger
from utils.metrics import (
    timing_decorator, time_operation, async_time_operation,
    log_performance, log_transcription_performance, log_tts_performance,
    log_tool_execution, get_metrics_collector, TimerContext
)

# Setup logging
setup_logging(level="DEBUG", format_type="standard", enable_colors=True)
logger = get_logger(__name__)

# Example 1: Basic logging usage
def example_basic_logging():
    """Demonstrate basic logging patterns"""
    logger.info("Starting example application")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    
    try:
        # Simulate some work
        time.sleep(0.1)
        logger.info("Work completed successfully")
    except Exception as e:
        logger.error(f"Error occurred: {e}")

# Example 2: Using timing decorator
@timing_decorator(operation="example_function", include_args=True)
def example_function_timing(text: str, delay: float = 0.1):
    """Function that demonstrates automatic timing"""
    logger.info(f"Processing: {text}")
    time.sleep(delay)
    return f"Processed: {text}"

# Example 3: Async timing decorator
@timing_decorator(operation="async_example")
async def async_example_function(duration: float = 0.1):
    """Async function with timing"""
    logger.info("Starting async work")
    await asyncio.sleep(duration)
    logger.info("Async work completed")
    return "Done"

# Example 4: Manual timing with context managers
def example_manual_timing():
    """Demonstrate manual timing patterns"""
    
    # Simple timing
    with time_operation("manual_operation", {"type": "example"}):
        time.sleep(0.05)
    
    # Complex timing with checkpoints
    timer = TimerContext("complex_operation", {"complexity": "high"})
    timer.start()
    
    time.sleep(0.02)
    timer.checkpoint("phase_1")
    
    time.sleep(0.03)
    timer.checkpoint("phase_2")
    
    time.sleep(0.01)
    total_time = timer.end()
    
    logger.info(f"Complex operation completed in {total_time:.2f}ms")

# Example 5: Specialized performance logging
def example_specialized_logging():
    """Demonstrate specialized performance logging functions"""
    
    # Transcription performance
    log_transcription_performance(
        audio_duration_ms=2000,     # 2 seconds of audio
        processing_duration_ms=250,  # 250ms to process
        confidence=0.95,
        text_length=50
    )
    
    # TTS performance
    log_tts_performance(
        text_length=100,
        generation_duration_ms=500,
        audio_size_bytes=24000,
        streaming=True
    )
    
    # Tool execution
    log_tool_execution(
        tool_name="spotify_play",
        duration_ms=150,
        success=True
    )

# Example 6: Async context manager
async def example_async_timing():
    """Demonstrate async timing patterns"""
    async with async_time_operation("async_operation", {"mode": "streaming"}):
        await asyncio.sleep(0.1)
        logger.info("Async operation in progress")

async def main():
    """Main example function"""
    logger.info("=" * 50)
    logger.info("Starting logging and metrics examples")
    logger.info("=" * 50)
    
    # Run examples
    example_basic_logging()
    
    result = example_function_timing("Hello World", 0.05)
    logger.info(f"Function result: {result}")
    
    await async_example_function(0.03)
    
    example_manual_timing()
    example_specialized_logging()
    await example_async_timing()
    
    # Show metrics summary
    collector = get_metrics_collector()
    logger.info("\n" + collector.get_summary_report())
    
    logger.info("=" * 50)
    logger.info("Examples completed")
    logger.info("=" * 50)

if __name__ == "__main__":
    asyncio.run(main()) 