"""
Configuration management for house-agent logging and metrics.

Provides centralized configuration for logging levels, formats, and performance
monitoring settings.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class LoggingConfig:
    """Configuration for logging system"""
    level: str = "INFO"
    format_type: str = "standard"  # standard, structured, minimal
    log_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_colors: bool = True
    
    @classmethod
    def from_env(cls) -> 'LoggingConfig':
        """Create configuration from environment variables"""
        return cls(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format_type=os.getenv("LOG_FORMAT", "standard").lower(),
            log_file=os.getenv("LOG_FILE"),
            max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", "10485760")),  # 10MB
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            enable_console=os.getenv("LOG_ENABLE_CONSOLE", "true").lower() == "true",
            enable_colors=os.getenv("LOG_ENABLE_COLORS", "true").lower() == "true"
        )

@dataclass
class MetricsConfig:
    """Configuration for metrics collection"""
    enabled: bool = True
    max_metrics: int = 10000
    enable_performance_logging: bool = True
    log_slow_operations_ms: float = 1000  # Log operations slower than this
    log_ttft: bool = True
    log_transcription: bool = True
    log_tts: bool = True
    log_tool_execution: bool = True
    
    @classmethod
    def from_env(cls) -> 'MetricsConfig':
        """Create configuration from environment variables"""
        return cls(
            enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            max_metrics=int(os.getenv("METRICS_MAX_COUNT", "10000")),
            enable_performance_logging=os.getenv("METRICS_LOG_PERFORMANCE", "true").lower() == "true",
            log_slow_operations_ms=float(os.getenv("METRICS_SLOW_THRESHOLD_MS", "1000")),
            log_ttft=os.getenv("METRICS_LOG_TTFT", "true").lower() == "true",
            log_transcription=os.getenv("METRICS_LOG_TRANSCRIPTION", "true").lower() == "true",
            log_tts=os.getenv("METRICS_LOG_TTS", "true").lower() == "true",
            log_tool_execution=os.getenv("METRICS_LOG_TOOLS", "true").lower() == "true"
        )

@dataclass
class HouseAgentConfig:
    """Main configuration for house-agent"""
    environment: str = "development"  # development, production, testing
    logging: LoggingConfig = None
    metrics: MetricsConfig = None
    
    def __post_init__(self):
        if self.logging is None:
            self.logging = LoggingConfig.from_env()
        if self.metrics is None:
            self.metrics = MetricsConfig.from_env()
    
    @classmethod
    def from_env(cls) -> 'HouseAgentConfig':
        """Create configuration from environment variables"""
        env = os.getenv("HOUSE_AGENT_ENV", "development").lower()
        
        config = cls(environment=env)
        
        # Environment-specific defaults
        if env == "production":
            config.logging.level = "INFO"
            config.logging.format_type = "structured"
            config.logging.enable_colors = False
            config.logging.log_file = "logs/house-agent-prod.log"
        elif env == "testing":
            config.logging.level = "WARNING"
            config.logging.format_type = "minimal"
            config.logging.enable_colors = False
            config.metrics.enabled = False
        else:  # development
            config.logging.level = "DEBUG"
            config.logging.format_type = "standard"
            config.logging.enable_colors = True
            config.logging.log_file = "logs/house-agent-dev.log"
        
        return config

# Global configuration instance
_global_config: Optional[HouseAgentConfig] = None

def get_config() -> HouseAgentConfig:
    """Get the global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = HouseAgentConfig.from_env()
    return _global_config

def set_config(config: HouseAgentConfig) -> None:
    """Set the global configuration instance"""
    global _global_config
    _global_config = config

def is_development() -> bool:
    """Check if running in development environment"""
    return get_config().environment == "development"

def is_production() -> bool:
    """Check if running in production environment"""
    return get_config().environment == "production"

def is_testing() -> bool:
    """Check if running in testing environment"""
    return get_config().environment == "testing"

def get_log_level() -> str:
    """Get the current log level"""
    return get_config().logging.level

def should_log_performance() -> bool:
    """Check if performance logging is enabled"""
    return get_config().metrics.enable_performance_logging

def get_slow_operation_threshold() -> float:
    """Get the threshold for logging slow operations"""
    return get_config().metrics.log_slow_operations_ms

# Environment variable documentation
ENV_VARS_DOCUMENTATION = """
House Agent Logging and Metrics Environment Variables:

=== Logging Configuration ===
LOG_LEVEL                   - Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT                  - Log format (standard, structured, minimal)
LOG_FILE                    - Log file path (optional)
LOG_MAX_FILE_SIZE          - Maximum log file size in bytes (default: 10485760)
LOG_BACKUP_COUNT           - Number of backup log files (default: 5)
LOG_ENABLE_CONSOLE         - Enable console logging (true/false, default: true)
LOG_ENABLE_COLORS          - Enable colored console output (true/false, default: true)

=== Metrics Configuration ===
METRICS_ENABLED            - Enable metrics collection (true/false, default: true)
METRICS_MAX_COUNT          - Maximum number of metrics to keep (default: 10000)
METRICS_LOG_PERFORMANCE    - Enable performance logging (true/false, default: true)
METRICS_SLOW_THRESHOLD_MS  - Threshold for logging slow operations in ms (default: 1000)
METRICS_LOG_TTFT          - Log Time To First Token metrics (true/false, default: true)
METRICS_LOG_TRANSCRIPTION - Log transcription metrics (true/false, default: true)
METRICS_LOG_TTS           - Log TTS metrics (true/false, default: true)
METRICS_LOG_TOOLS         - Log tool execution metrics (true/false, default: true)

=== Environment ===
HOUSE_AGENT_ENV           - Environment mode (development, production, testing)

Examples:
  # Development with debug logging
  export HOUSE_AGENT_ENV=development
  export LOG_LEVEL=DEBUG
  
  # Production with structured logging
  export HOUSE_AGENT_ENV=production
  export LOG_FORMAT=structured
  export LOG_FILE=/var/log/house-agent.log
  
  # Testing with minimal logging
  export HOUSE_AGENT_ENV=testing
  export METRICS_ENABLED=false
""" 