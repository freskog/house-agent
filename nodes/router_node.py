"""
Router node implementation.

This router uses unified routing logic with DistilBERT-based intent detection.
"""

from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage
from .base_node import BaseNode
from .schemas import AgentState
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Lazy import for sentence transformers to avoid startup delay
_sentence_transformer_model = None

def get_sentence_transformer():
    """Lazy loading of sentence transformer model."""
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model for intent classification")
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to keyword matching")
            _sentence_transformer_model = False
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            _sentence_transformer_model = False
    return _sentence_transformer_model if _sentence_transformer_model is not False else None

class RouterNode(BaseNode):
    """
    Router with unified interface and DistilBERT-based intent detection.
    
    Uses confidence thresholds and domain detection for fast, reliable routing.
    """
    
    def __init__(self):
        """Initialize router with embedding-based intent classification."""
        super().__init__([], "Router")
        self.confidence_threshold = 0.30  # Lowered threshold for better routing to specialists
        
        # Domain-specific example phrases for embedding comparison
        self.domain_examples = {
            "music": [
                "play some music", "turn up the volume", "next song", "pause the music", 
                "play spotify", "stop the music", "previous track", "what's playing",
                "shuffle playlist", "play jazz", "turn down volume", "skip song",
                "resume music", "change song", "music volume up", "mute music",
                "play my playlist", "start music", "music control", "audio playback"
            ],
            "house": [
                "turn on the lights", "dim the lights", "adjust temperature", "turn off the lights",
                "make it warmer", "cool down the room", "brighten the lights", "heat the house",
                "turn on air conditioning", "set temperature", "lighting control", "smart home",
                "lights on", "lights off", "temperature control", "room lighting",
                "home automation", "climate control", "light dimming", "thermostat"
            ],
            "search": [
                "what's the weather", "search for news", "who won the game", "tell me about",
                "find information", "what happened today", "look up", "weather forecast",
                "news update", "sports scores", "search the web", "get information about",
                "weather today", "latest news", "search results", "information query",
                "web search", "find out about", "research topic", "current events",
                "when is the next", "tournament schedule", "event dates", "competition results",
                "who is the winner", "championship information", "sports events", "game schedule",
                "latest scores", "tournament bracket", "league standings", "match results"
            ]
        }
        
        # Will be computed lazily when first needed
        self._domain_embeddings = None
        self._use_embeddings = True
    
    async def handle_request(self, state: AgentState) -> AgentState:
        """
        Route requests using simplified logic.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with routing decision
        """
        user_message = self.get_last_user_message(state)
        if not user_message:
            return self.create_response(
                [AIMessage(content="I didn't receive a clear request.")],
                next_action="end"
            )
        
        # Classify intent using simplified logic
        intent = await self.classify_intent(user_message.content)
        
        # Route based on intent confidence and domain count
        if intent["confidence"] > self.confidence_threshold and len(intent["domains"]) == 1:
            # High confidence, single domain -> route to specialist
            target_domain = intent["domains"][0]
            return {
                **state,
                "next_action": "continue",
                "target_domain": target_domain
            }
        else:
            # Low confidence or multiple domains -> escalate to agent
            return {
                **state,
                "next_action": "escalate", 
                "target_domain": "agent"
            }
    
    def _compute_domain_embeddings(self):
        """Compute and cache domain embeddings."""
        model = get_sentence_transformer()
        if not model:
            logger.warning("Sentence transformer not available, using keyword fallback")
            self._use_embeddings = False
            return
            
        self._domain_embeddings = {}
        for domain, examples in self.domain_examples.items():
            try:
                embeddings = model.encode(examples)
                # Use mean embedding as domain representation
                self._domain_embeddings[domain] = np.mean(embeddings, axis=0)
                logger.debug(f"Computed embeddings for {domain} domain ({len(examples)} examples)")
            except Exception as e:
                logger.error(f"Failed to compute embeddings for {domain}: {e}")
                self._use_embeddings = False
                return
                
        logger.info(f"Successfully computed embeddings for {len(self._domain_embeddings)} domains")

    async def classify_intent(self, user_request: str) -> Dict[str, Any]:
        """
        Classify user intent using DistilBERT embeddings with keyword fallback.
        
        Args:
            user_request: User's request text
            
        Returns:
            Dictionary with domains, confidence, and classification
        """
        # Try embedding-based classification first
        if self._use_embeddings:
            try:
                return await self._classify_with_embeddings(user_request)
            except Exception as e:
                logger.error(f"Embedding classification failed: {e}")
                logger.info("Falling back to keyword-based classification")
                self._use_embeddings = False
        
        # Fallback to keyword-based classification
        return await self._classify_with_keywords(user_request)
    
    async def _classify_with_embeddings(self, user_request: str) -> Dict[str, Any]:
        """Classify using sentence embeddings."""
        # Lazy load domain embeddings
        if self._domain_embeddings is None:
            self._compute_domain_embeddings()
            
        if not self._domain_embeddings:
            raise Exception("No domain embeddings available")
            
        model = get_sentence_transformer()
        if not model:
            raise Exception("Sentence transformer model not available")
        
        # Encode user request
        request_embedding = model.encode([user_request])
        
        # Calculate similarities to each domain
        similarities = {}
        for domain, domain_embedding in self._domain_embeddings.items():
            similarity = cosine_similarity(request_embedding, [domain_embedding])[0][0]
            similarities[domain] = float(similarity)
        
        # Find best match and determine confidence
        best_domain = max(similarities, key=similarities.get)
        confidence = similarities[best_domain]
        
        # Determine detected domains (similarity threshold)
        similarity_threshold = 0.35  # Optimized to reduce multi-domain false positives
        detected_domains = [domain for domain, sim in similarities.items() if sim > similarity_threshold]
        
        # If no domains meet threshold, use the best one if confidence is reasonable
        if not detected_domains and confidence > 0.2:
            detected_domains = [best_domain]
        elif not detected_domains:
            detected_domains = ["clarification"]
            confidence = 0.0
        
        return {
            "domains": detected_domains,
            "confidence": confidence,
            "scores": similarities,
            "complexity": "simple" if len(detected_domains) <= 1 else "complex"
        }
    
    async def _classify_with_keywords(self, user_request: str) -> Dict[str, Any]:
        """Fallback keyword-based classification (original implementation)."""
        request_lower = user_request.lower()
        detected_domains = []
        confidence_scores = {}
        
        # Music domain detection
        music_keywords = ["play", "music", "song", "volume", "spotify", "pause", "stop", "next", "previous"]
        music_score = sum(1 for keyword in music_keywords if keyword in request_lower)
        if music_score > 0:
            detected_domains.append("music")
            confidence_scores["music"] = min(music_score / 2.0, 1.0)
        
        # House domain detection
        house_keywords = ["light", "lights", "temperature", "heat", "cool", "turn on", "turn off", "dim", "bright"]
        house_score = sum(1 for keyword in house_keywords if keyword in request_lower)
        if house_score > 0:
            detected_domains.append("house")
            confidence_scores["house"] = min(house_score / 3.0, 1.0)
        
        # Search domain detection
        search_keywords = ["search", "find", "weather", "news", "what is", "who is", "when is", "where is", "who won", "what happened", "tell me about"]
        search_score = sum(1 for keyword in search_keywords if keyword in request_lower)
        if search_score > 0:
            detected_domains.append("search")
            confidence_scores["search"] = min(search_score / 1.5, 1.0)
        
        # Calculate overall confidence
        if not detected_domains:
            overall_confidence = 0.0
            detected_domains = ["clarification"]
        elif len(detected_domains) == 1:
            overall_confidence = confidence_scores[detected_domains[0]]
        else:
            overall_confidence = 0.5
        
        return {
            "domains": detected_domains,
            "confidence": overall_confidence,
            "scores": confidence_scores,
            "complexity": "simple" if len(detected_domains) <= 1 else "complex"
        }
    
    def should_handle_request(self, message: str) -> bool:
        """Router handles all initial requests."""
        return True  # Router is always the entry point
    
    def get_route_destination(self, state: AgentState) -> str:
        """
        Get routing destination for backward compatibility.
        
        This method is kept for compatibility with existing graph structure
        while we transition to the simplified routing.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node destination
        """
        next_action = state.get("next_action")
        target_domain = state.get("target_domain")
        
        if next_action == "escalate":
            return "agent"
        elif next_action == "continue" and target_domain:
            return target_domain
        else:
            return "agent"  # Default fallback 