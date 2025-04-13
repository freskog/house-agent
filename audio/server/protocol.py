"""
Protocol definitions for audio server communication.

This module defines the message types and structures used for communication
between the edge device and the server.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
import json
import time
from pydantic import BaseModel, Field

class MessageType(str, Enum):
    """Types of messages that can be exchanged"""
    AUDIO_STREAM = "audio_stream"
    AUDIO_PLAYBACK = "audio_playback"
    STATUS = "status"
    ERROR = "error"

# Standardized audio configuration - no more options for format
class AudioConfig(BaseModel):
    """Audio configuration parameters"""
    # Standard configuration for WhisperCPP: 16kHz, 16-bit, mono
    channels: int = 1
    sample_rate: int = 16000
    chunk_size: int = 2048

class AudioStreamPayload(BaseModel):
    """Payload for audio stream messages"""
    # Always RAW PCM, 16-bit, mono, 16kHz
    audio: bytes

class AudioPlaybackPayload(BaseModel):
    """Payload for audio playback messages"""
    # Always WAV format - this is what the TTS engine produces
    audio: bytes

class StatusPayload(BaseModel):
    """Payload for status messages"""
    is_speech: bool
    confidence: float
    timestamps: Optional[List[Dict[str, Any]]] = None

class ErrorPayload(BaseModel):
    """Payload for error messages"""
    error: str
    code: Optional[int] = None

class Message(BaseModel):
    """Base message structure"""
    type: MessageType
    timestamp: float = Field(default_factory=time.time)
    sequence: int
    payload: Dict[str, Any]

    def to_json(self) -> str:
        """Convert message to JSON string"""
        return self.json()

    @classmethod
    def from_json(cls, data: str) -> 'Message':
        """Create message from JSON string"""
        return cls.parse_raw(data)

    @classmethod
    def create_audio_stream(cls, sequence: int, audio_data: bytes) -> 'Message':
        """Create an audio stream message"""
        return cls(
            type=MessageType.AUDIO_STREAM,
            sequence=sequence,
            payload={
                "audio": audio_data
            }
        )

    @classmethod
    def create_audio_playback(cls, sequence: int, audio_data: bytes) -> 'Message':
        """Create an audio playback message"""
        return cls(
            type=MessageType.AUDIO_PLAYBACK,
            sequence=sequence,
            payload={
                "audio": audio_data
            }
        )

    @classmethod
    def create_status(cls, sequence: int, is_speech: bool, confidence: float, timestamps: Optional[List[Dict[str, Any]]] = None) -> 'Message':
        """Create a status message"""
        return cls(
            type=MessageType.STATUS,
            sequence=sequence,
            payload={
                "is_speech": is_speech,
                "confidence": confidence,
                "timestamps": timestamps
            }
        )

    @classmethod
    def create_error(cls, sequence: int, error: str, code: Optional[int] = None) -> 'Message':
        """Create an error message"""
        return cls(
            type=MessageType.ERROR,
            sequence=sequence,
            payload={
                "error": error,
                "code": code
            }
        )

# Default audio configuration
DEFAULT_AUDIO_CONFIG = AudioConfig() 