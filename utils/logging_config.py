"""
Centralized logging configuration for house-agent.

Provides standardized logging setup with different configurations for
development and production environments.
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum

class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogFormat(Enum):
    """Log format options"""
    STANDARD = "standard"
    STRUCTURED = "structured"
    MINIMAL = "minimal"

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured (JSON) logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add thread info if available
        if hasattr(record, 'thread_id'):
            log_data["thread_id"] = record.thread_id
            
        # Add performance metrics if available
        if hasattr(record, 'duration_ms'):
            log_data["duration_ms"] = record.duration_ms
        if hasattr(record, 'operation'):
            log_data["operation"] = record.operation
            
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                if not key.startswith('_'):
                    log_data[key] = value
                    
        return json.dumps(log_data)

class ColoredFormatter(logging.Formatter):
    """Formatter with color support for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors if terminal supports it."""
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
            
        return super().format(record)

def setup_logging(
    name: str = None,
    level: str = "INFO",
    format_type: str = "standard",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
    enable_colors: bool = True,
    structured_fields: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        name: Logger name (if provided, returns named logger; if None, configures root)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type (standard, structured, minimal)
        log_file: Optional log file path
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files to keep
        enable_console: Whether to enable console logging
        enable_colors: Whether to enable colored console output
        structured_fields: Additional fields for structured logging
        
    Returns:
        Configured logger (named if name provided, root otherwise)
    """
    # If name is provided and looks like a log level, it's being used incorrectly
    if name and name.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        # Swap the parameters - name was passed as level
        level = name
        name = None
    
    # If name is provided, return a named logger after ensuring root is configured
    if name:
        if not logging.getLogger().handlers:
            # Configure root logger first
            setup_logging(name=None, level=level, format_type=format_type, log_file=log_file,
                         max_file_size=max_file_size, backup_count=backup_count, 
                         enable_console=enable_console, enable_colors=enable_colors,
                         structured_fields=structured_fields)
        return logging.getLogger(name)
    
    # Configure root logger
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Already configured
        return root_logger
        
    # Set log level
    log_level = getattr(logging, level.upper())
    root_logger.setLevel(log_level)
    
    # Create formatters based on format type
    if format_type == "structured":
        formatter = StructuredFormatter()
    elif format_type == "minimal":
        formatter = logging.Formatter('%(levelname)s: %(message)s')
    else:  # standard
        if enable_colors and enable_console:
            formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    # Setup console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        
        # Use structured format for file logging if specified
        if format_type == "structured":
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def configure_for_development():
    """Configure logging for development environment."""
    return setup_logging(
        level="DEBUG",
        format_type="standard",
        enable_console=True,
        enable_colors=True,
        log_file="logs/house-agent-dev.log"
    )

def configure_for_production():
    """Configure logging for production environment."""
    return setup_logging(
        level="INFO",
        format_type="structured",
        enable_console=True,
        enable_colors=False,
        log_file="logs/house-agent-prod.log"
    )

def configure_for_testing():
    """Configure logging for testing environment."""
    return setup_logging(
        level="WARNING",
        format_type="minimal",
        enable_console=True,
        enable_colors=False
    )

# Auto-configure based on environment
def auto_configure():
    """Automatically configure logging based on environment variables."""
    env = os.getenv("HOUSE_AGENT_ENV", "development").lower()
    
    if env == "production":
        return configure_for_production()
    elif env == "testing":
        return configure_for_testing()
    else:
        return configure_for_development()

# Initialize logging if this module is imported
if not logging.getLogger().handlers:
    auto_configure() 