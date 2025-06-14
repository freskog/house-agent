from typing import Optional
from .tts import TTSEngine

# Import our logging infrastructure
from utils.logging_config import setup_logging

# Set up logger for this module
logger = setup_logging(__name__)

class CallbackRegistry:
    """Registry for callbacks"""
    def __init__(self):
        self.callbacks = {}
        
    def register(self, name, callback):
        """Register a callback"""
        self.callbacks[name] = callback
        
    def get(self, name):
        """Get a callback by name"""
        return self.callbacks.get(name)

class AudioServer:
    """Server for audio processing (transcription, TTS, etc.)"""
    def __init__(self, 
                 config_path: Optional[str] = None, 
                 debug: bool = False):
        """Initialize the audio server
        
        Args:
            config_path: Path to configuration file
            debug: Enable debug logging
        """
        self.debug = debug
        self.log("Initializing AudioServer")
        
        # Initialize engines
        self.stt_engine = None
        self.tts_engine = TTSEngine()  # Initialize with defaults
        self.log(f"TTS engine initialized: {self.tts_engine.is_initialized()}")
        
        # Initialize the callback registry
        self.callback_registry = CallbackRegistry()
        
        # Register default callbacks
        self.register_default_callbacks()
        
        # Dictionary to store client thread IDs
        self.client_threads = {}
        
    def log(self, message):
        """Log a message"""
        if self.debug:
            logger.debug(f"[AudioServer] {message}")
        
    def register_default_callbacks(self):
        """Register default callbacks"""
        self.callback_registry.register("demo", self.demo_transcription_callback)
    
    def get_client_thread_id(self, client_id):
        """Get or create a thread ID for a client
        
        Args:
            client_id: Unique identifier for the client
            
        Returns:
            str: Thread ID for the client
        """
        if client_id not in self.client_threads:
            # Create a new thread ID for this client
            import uuid
            self.client_threads[client_id] = str(uuid.uuid4())
            self.log(f"Created new thread ID for client: {self.client_threads[client_id]}")
        
        return self.client_threads[client_id]
        
    async def demo_transcription_callback(self, transcription):
        """Demo callback that echoes what was heard"""
        return f"I heard: {transcription.text}"
        
    async def text_to_speech(self, text: str, voice: Optional[str] = None) -> bytes:
        """Convert text to speech
        
        Args:
            text: Text to convert to speech
            voice: Voice to use for speech
            
        Returns:
            bytes: Audio data in WAV format
        """
        if not text:
            self.log("Empty text provided for TTS")
            return b''
        
        self.log(f"Converting text to speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Process the full text through TTS engine
        audio_data = await self.tts_engine.synthesize(text)
        
        if not audio_data:
            self.log("Failed to synthesize speech")
            return b''
            
        self.log(f"Generated {len(audio_data)} bytes of audio")
        return audio_data 