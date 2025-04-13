"""
Audio package for voice interaction.

This package provides components for:
- Voice Activity Detection (VAD)
- Speech-to-Text transcription 
- Text-to-Speech synthesis
- WebSocket audio server
"""

__version__ = "0.1.0"

from .server import (
    AudioServer,
    Message,
    MessageType,
    AudioConfig,
    VADConfig,
    TranscriptionConfig,
    TranscriptionResult
)

__all__ = [
    'AudioServer',
    'Message',
    'MessageType',
    'AudioConfig',
    'VADConfig',
    'TranscriptionConfig',
    'TranscriptionResult'
] 