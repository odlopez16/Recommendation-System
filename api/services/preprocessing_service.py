from typing import Any
import spacy
import langid
from functools import lru_cache
import logging


MODELS: dict[str, Any] = {
    'en': spacy.load('en_core_web_sm'),
    'es': spacy.load('es_core_news_sm')
}

logger = logging.getLogger("api.services.preprocessing_service")

class TextPreprocessor:
    def __init__(self):
        self.__min_text_length = 5
        logger.debug("TextPreprocessor initialized with min_text_length=%d", self.__min_text_length)

    @lru_cache(maxsize=1000)
    def preprocess(self, text: str)-> str:
        if len(text.strip()) < self.__min_text_length:
            logger.warning("Text too short to preprocess: '%s'", text)
            return ""
        
        language, _ = langid.classify(text)
        logger.info("Detected language: %s", language)

        if language == 'en':
            nlp = MODELS.get(language, MODELS['en'])
        else:
            nlp = MODELS.get(language, MODELS['es'])

        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        logger.debug("Preprocessed tokens: %s", tokens)
        return " ".join(tokens)