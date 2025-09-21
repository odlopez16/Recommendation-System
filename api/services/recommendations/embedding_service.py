import sys
from pathlib import Path

from fastapi import HTTPException, status
from pydantic import UUID4
from api.models.user_interaction_model import UserProductInteractionInDB

sys.path.append(str(Path(__file__).resolve().parent.parent))
import faiss #type: ignore
from typing import Any, Optional
from sqlalchemy import Insert, Select, desc, func
from api.models.embedding_model import Embedding, EmbeddingIn
from api.models.products_model import Product
from api.schemas.embedding_schema import embeddings_table
from api.schemas.product_schema import products_table
from api.database.database_config import primary_database as db
from api.schemas.user_interaction_schema import user_interactions_table
from api.models.auth_model import UserWithoutPassword
import numpy as np
from openai import OpenAI
from api.services.recommendations.preprocessing_service import TextPreprocessor
from api.services.cache_service import cache_manager
from config import config
import logging
import asyncio
from functools import lru_cache


class EmbeddingProcessor:
    """
    Handles the generation, storage, and retrieval of embeddings for products.
    """
    
    def __init__(self, openai_client: OpenAI, dimension: int = 768) -> None:
        self.logger = logging.getLogger("api.helpers.embedding.EmbeddingProcessor")
        self.__openai_client = openai_client
        self.__dimension = dimension
        self._text_preprocessor: Optional[TextPreprocessor] = None

    @property
    def text_preprocessor(self) -> TextPreprocessor:
        """Lazy loading del preprocessor"""
        if self._text_preprocessor is None:
            self._text_preprocessor = TextPreprocessor()
        return self._text_preprocessor
    
    async def generate_embeddings(self, text: str = "", products: list[Product] = []) -> list[tuple[Any, list[float]]] | Any:
        """
        Generate embeddings with batch processing and caching.
        """
        try:
            if text == "":
                return await self._generate_batch_embeddings(products)
            else:
                return await self._generate_single_embedding(text)
        except Exception as e:
            self.logger.error(f"Error generating embeddings")
            raise
    
    async def _generate_single_embedding(self, text: str) -> list[float]:
        """Generate single embedding with preprocessing"""
        processed_text = self.text_preprocessor.preprocess(text)
        embedding = self.__openai_client.embeddings.create(
            input=processed_text,
            model=config.OPENAI_MODEL,
            dimensions=self.__dimension
        ).data[0].embedding
        self.logger.info("Generated embedding for input text.")
        return embedding
    
    async def _generate_batch_embeddings(self, products: list[Product] = []) -> list[tuple[Any, list[float]]]:
        """Generate embeddings in batches for better performance"""
        if not products:
            from api.services.recommendations.product_service import product_processor
            products = await product_processor.get_products_from_primary_db(limit=1000)
        
        batch_size = 10
        embed_list: list[tuple[Any, list[float]]] = []
        
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            # Combine name and description for better semantic representation
            batch_texts = [self.text_preprocessor.preprocess(f"{prod.name} {prod.description}") for prod in batch]

            response = self.__openai_client.embeddings.create(
                input=batch_texts,
                model=config.OPENAI_MODEL,
                dimensions=self.__dimension
            )
            
            for j, embedding_data in enumerate(response.data):
                embed_list.append((batch[j].id, embedding_data.embedding))
            
            await asyncio.sleep(0.1)
        
        self.logger.info(f"Generated embeddings for products in batches.")
        return embed_list

    async def get_embedding_by_prod_id(self, product_id: Any) -> Embedding | None:
        """
        Retrieve an embedding from the database by product ID.
        Args:
            product_id (Any): The product ID.
        Returns:
            Embedding: The embedding object or None if not found.
        """
        self.logger.debug(f"Fetching embedding")
        query: Select = embeddings_table.select().where(embeddings_table.c.product_id == product_id)
        record = await db.get_database().fetch_one(query)
        if record is not None:
            embedding: Embedding = Embedding(
                id=record["id"],
                product_id=record["product_id"],
                embedding=np.frombuffer(record["embedding"], dtype=np.float32).tolist(),
                created_at=record["created_at"]
                )
            return embedding
        else:
            return None 

    async def save_embeddings(self, embeddings_generated: list[tuple[Any, list[float]]]) -> None:
        try:
            saved_count = 0
            for pair in embeddings_generated:
                # Check if embedding already exists
                existing = await self.get_embedding_by_prod_id(pair[0])
                if existing is not None:
                    continue
                
                embedding_bytes = np.array(pair[1], dtype=np.float32).tobytes()
                query: Insert = embeddings_table.insert().values(
                    product_id=pair[0], 
                    embedding=embedding_bytes
                )
                await db.get_database().execute(query)
                saved_count += 1
            self.logger.info(f"Saved {saved_count} new embeddings to the database.")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {str(e)}")
            raise e

    async def get_embeddings_from_db(self) -> list[Embedding]:
        """
        Retrieve embeddings from database.
        """
        try:
            query: Select = embeddings_table.select()
            records = await db.get_database().fetch_all(query)
            
            embeddings: list[Embedding] = [Embedding(
                        id=record["id"],
                        product_id=record["product_id"],
                        embedding=np.frombuffer(record["embedding"], dtype=np.float32).tolist(),
                        created_at=record["created_at"]
                    ) for record in records]
            
            self.logger.info(f"Fetched {len(embeddings)} embeddings from database.")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error fetching embeddings from database: {str(e)}")
            return []

    def get_embeddings_average(self, embed_list: list[list[float]]) -> np.ndarray | Any:
        """
        Calculate the average of a list of embeddings.
        Args:
            embed_list (list[Embedding]): List of Embedding objects.
        Returns:
            np.ndarray[Float32DType, Any] | []: The average of the embeddings.
        """
        if not embed_list:
            return []

        average_embedding = [sum([embed_list[i][j] 
        for i in range(len(embed_list))]) / len(embed_list) 
        for j in range(self.__dimension)]
        return np.array(average_embedding, dtype=np.float32)

class FaissManager:
    """
    Optimized FAISS manager with lazy initialization and caching.
    """
    _instance: Optional['FaissManager'] = None
    _index: Optional[faiss.Index] = None
    
    def __new__(cls, embed_list: list[Embedding], dimension: int = 768):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, embed_list: list[Embedding], dimension: int = 768):
        if hasattr(self, '_initialized'):
            return
        self.logger = logging.getLogger("api.helpers.embedding.FaissManager")
        self.embed_list = embed_list
        self.dimension = dimension
        self._initialized = True
        self.logger.info(f"Initialized FAISS manager.")
    
    @property
    def index(self) -> faiss.Index:
        """Lazy initialization of FAISS index"""
        if self._index is None:
            self._index = faiss.IndexFlatL2(self.dimension)
            self.logger.info("Created FAISS index on demand.")
        return self._index
    
    @property
    def embeddings(self) -> np.ndarray:
        """Lazy loading of embeddings array"""
        if not hasattr(self, '_embeddings_array'):
            self._embeddings_array = np.array([e.embedding for e in self.embed_list], dtype=np.float32)
        return self._embeddings_array

    def update_index(self) -> None:
        """
        Add embeddings to FAISS index with optimization.
        """
        if len(self.embed_list) == 0:
            self.logger.warning("No embeddings to add to index.")
            return
            
        embeddings_array = self.embeddings
        if embeddings_array.size == 0:
            self.logger.warning("Empty embeddings array.")
            return
            
        if not self.index.is_trained:
            self.logger.debug("Training FAISS index.")
            self.index.train(embeddings_array) #type: ignore
        
        self.index.add(embeddings_array) #type: ignore
        self.logger.info(f"Added {len(self.embed_list)} embeddings to FAISS index.")

    async def search(self, query: np.ndarray, k: int | None = None) -> list[Product]:
        """Optimized search with batch product fetching"""
        # Default to 50 results if k is None, or use all available if less than 50
        if k is None:
            search_k = min(50, self.index.ntotal)
        else:
            search_k = min(k, self.index.ntotal)
            
        self.logger.debug(f"Searching FAISS index for {search_k} results.")
        
        if self.index.ntotal == 0:
            self.logger.warning("FAISS index is empty.")
            return []
            
        if query.ndim == 1:
            query = query.reshape(1, -1).astype(np.float32)
            
        try:
            distances, indexes = self.index.search(query, search_k) #type: ignore
            
            # Create list of (distance, product) pairs to ensure proper sorting
            product_distance_pairs = []
            from api.services.recommendations.product_service import product_processor
            
            for i, idx in enumerate(indexes[0]):
                if 0 <= idx < len(self.embed_list):
                    try:
                        product = await product_processor.get_product_by_id(self.embed_list[idx].product_id)
                        product_distance_pairs.append((distances[0][i], product))
                    except Exception as e:
                        self.logger.warning(f"Could not fetch product {self.embed_list[idx].product_id}: {e}")
                        continue
            
            # Sort by distance (lower = more similar) and extract products
            product_distance_pairs.sort(key=lambda x: x[0])
            recommended_products = [product for _, product in product_distance_pairs]
            
            self.logger.info(f"Found {len(recommended_products)} recommended products sorted by similarity")
            return recommended_products
            
        except Exception as e:
            self.logger.error(f"Error in FAISS search: {str(e)}")
            return []