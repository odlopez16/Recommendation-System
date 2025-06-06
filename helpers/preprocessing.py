from typing import Any
import spacy
import langid
from functools import lru_cache


MODELS: dict[str, Any] = {
    'en': spacy.load('en_core_web_sm'),
    'es': spacy.load('es_core_news_sm')
}

class TextPreprocessor:
    def __init__(self):
        self.min_text_length = 5

    @lru_cache(maxsize=1000)
    def preprocess(self, text: str)-> str:
        if len(text.strip()) < self.min_text_length:
            return ""
        
        language, _ = langid.classify(text)

        if language == 'en':
            nlp = MODELS.get(language, MODELS['en'])
        nlp = MODELS.get(language, MODELS['es'])

        doc = nlp(text.lower())
        return " ".join([
            token.lemma_ for token in doc
            if not token.is_stop and token.is_alpha
        ])