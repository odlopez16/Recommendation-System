from typing import Any
from functools import lru_cache
import logging
import spacy
import langid # type: ignore
import psutil # type: ignore


MODELS: dict[str, Any] = {
    'en': spacy.load('en_core_web_sm'),  
    'es': spacy.load('es_core_news_sm')  
}

# Configuración del logger para este servicio
logger = logging.getLogger("api.services.preprocessing_service")

class TextPreprocessor:
    """
    Clase para preprocesar texto usando modelos de lenguaje natural.
    
    Esta clase maneja el preprocesamiento de texto para diferentes idiomas,
    incluyendo la detección automática del idioma, lematización y eliminación
    de stop words. También incluye gestión de memoria y caché para optimizar
    el rendimiento.
    
    Attributes:
        __min_text_length (int): Longitud mínima del texto para ser procesado
        __memory_threshold (int): Porcentaje máximo de memoria antes de limpiar caché
    """
    def __init__(self):
        """
        Inicializa el preprocesador de texto con valores predeterminados.
        
        El preprocesador se configura con una longitud mínima de texto de 5 caracteres
        y un umbral de memoria del 80% para la limpieza automática del caché.
        """
        self.__min_text_length = 5
        self.__memory_threshold = 80
        logger.debug("TextPreprocessor initialized with min_text_length")

    @lru_cache(maxsize=1000)
    def preprocess(self, text: str)-> str:
        """
        Preprocesa un texto aplicando varios pasos de procesamiento de lenguaje natural.
        
        El método realiza los siguientes pasos:
        1. Verifica la longitud mínima del texto
        2. Detecta el idioma automáticamente
        3. Aplica lematización usando el modelo correspondiente al idioma
        4. Elimina stop words y caracteres no alfabéticos
        
        Args:
            text (str): El texto a preprocesar
            
        Returns:
            str: Texto preprocesado con tokens lematizados separados por espacios
                o cadena vacía si el texto es muy corto
        
        Note:
            Los resultados se almacenan en caché para mejorar el rendimiento
            en textos repetidos.
        """
        if len(text.strip()) < self.__min_text_length:
            logger.warning("Text too short to preprocess")
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

    def check_memory_usage(self):
        """
        Verifica el uso de memoria y limpia el caché si es necesario.
        
        Este método monitorea el uso de memoria del sistema y, si supera
        el umbral configurado (__memory_threshold), limpia el caché de
        preprocesamiento para liberar memoria.
        
        Note:
            Este método es útil para prevenir problemas de memoria en
            sistemas con recursos limitados o bajo alta carga.
        """
        memory = psutil.virtual_memory()
        if memory.percent > self.__memory_threshold:
            self._clear_cache()
            logger.info(f"Cache limpiado por alto uso de memoria usado)")

    def _clear_cache(self):
        """
        Método privado para limpiar el caché de la función preprocess.
        
        Elimina todas las entradas almacenadas en el caché del método
        preprocess, liberando la memoria utilizada por los resultados
        previamente calculados.
        
        Note:
            Este método es llamado automáticamente por check_memory_usage
            cuando el uso de memoria supera el umbral establecido.
        """
        self.preprocess.cache_clear()
        logger.info("Cache de preprocess limpiado")