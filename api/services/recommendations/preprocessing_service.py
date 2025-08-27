from typing import Any, Optional
from functools import lru_cache
import logging
import spacy
import langid
import psutil
import threading

# Lazy loading de modelos para reducir tiempo de inicio
_models_cache: dict[str, Any] = {}
_models_lock = threading.Lock()

def get_spacy_model(language: str) -> Any:
    """Lazy loading de modelos spaCy"""
    with _models_lock:
        if language not in _models_cache:
            try:
                if language == 'en':
                    _models_cache[language] = spacy.load('en_core_web_sm')
                else:
                    _models_cache[language] = spacy.load('es_core_news_sm')
                logging.getLogger("preprocessing").info(f"Loaded spaCy model for {language}")
            except OSError:
                # Fallback to basic model
                _models_cache[language] = spacy.blank(language)
                logging.getLogger("preprocessing").warning(f"Using blank model for {language}")
        return _models_cache[language]

# Configuración del logger para este servicio
logger = logging.getLogger("api.services.preprocessing_service")

class TextPreprocessor:
    """
    Optimized text preprocessor with intelligent caching and lazy loading.
    """
    _instance: Optional['TextPreprocessor'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self.__min_text_length = 5
        self.__memory_threshold = 80
        self._language_cache = {}
        self._initialized = True
        logger.debug("TextPreprocessor initialized as singleton")

    @lru_cache(maxsize=2000)  # Increased cache size
    def preprocess(self, text: str) -> str:
        """
        Optimized text preprocessing with caching and fast language detection.
        """
        if len(text.strip()) < self.__min_text_length:
            return ""
        
        # Fast language detection with caching
        text_hash = hash(text[:100])  # Use first 100 chars for language detection
        if text_hash in self._language_cache:
            language = self._language_cache[text_hash]
        else:
            language, confidence = langid.classify(text)
            if confidence > 0.8:  # Only cache high-confidence detections
                self._language_cache[text_hash] = language
        
        # Use appropriate model
        nlp = get_spacy_model('en' if language == 'en' else 'es')
        
        # Optimized processing
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc 
                 if not token.is_stop and token.is_alpha and len(token.text) > 2]
        
        return " ".join(tokens)

    def check_memory_usage(self):
        """
        Smart memory management with selective cache clearing.
        """
        try:
            memory = psutil.virtual_memory()
            if memory.percent > self.__memory_threshold:
                # Clear language cache first (smaller impact)
                if len(self._language_cache) > 100:
                    self._language_cache.clear()
                    logger.info("Language cache cleared")
                
                # If still high memory, clear preprocessing cache
                if memory.percent > self.__memory_threshold + 5:
                    self._clear_cache()
                    logger.info(f"Full cache cleared - memory usage: {memory.percent}%")
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")

    def _clear_cache(self):
        """
        Clear preprocessing cache and language cache.
        """
        self.preprocess.cache_clear()
        self._language_cache.clear()
        logger.info("All caches cleared")
    
    def get_cache_info(self) -> dict:
        """Get cache statistics for monitoring"""
        return {
            'preprocess_cache': self.preprocess.cache_info()._asdict(),
            'language_cache_size': len(self._language_cache)
        }