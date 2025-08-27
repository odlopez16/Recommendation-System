import sys
from pathlib import Path

from pydantic import UUID4
from api.models.user_interaction_model import UserProductInteractionInDB
sys.path.append(str(Path(__file__).resolve().parent.parent))
import faiss
from typing import Any, Optional
from sqlalchemy import Insert, Select, desc, func
from api.models.embedding_model import Embedding, EmbeddingIn
from api.models.products_model import Product
from api.schemas.embedding_schema import embeddings_table
from api.schemas.product_schema import products_table
from api.database.database_config import primary_database as db
from api.schemas.user_interaction_schema import user_interactions_table
from api.services.recommendations.like_service import like_service
from api.models.auth_model import UserWithoutPassword
import numpy as np
from openai import OpenAI
from api.services.recommendations.preprocessing_service import TextPreprocessor
from api.services.recommendations.product_service import product_processor
from api.services.cache_service import cache_manager
from config import config
import logging
import asyncio
from functools import lru_cache


class EmbeddingProcessor:
    """
    Handles the generation, storage, and retrieval of embeddings for products.
    Optimized with caching, lazy loading, and batch processing.
    """
    _instance: Optional['EmbeddingProcessor'] = None
    _embeddings_cache: Optional[list[Embedding]] = None
    _text_preprocessor: Optional[TextPreprocessor] = None
    
    def __new__(cls, openai_client: OpenAI, dimension: int = 768):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, openai_client: OpenAI, dimension: int = 768) -> None:
        if hasattr(self, '_initialized'):
            return
        self.logger = logging.getLogger("api.helpers.embedding.EmbeddingProcessor")
        self.__openai_client = openai_client
        self.__dimension = dimension
        self.__like_service = like_service
        self._initialized = True

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
            self.logger.error(f"Error generating embeddings: {e}")
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
            products = await product_processor.get_products_from_primary_db(limit=50)  # Limit initial load
        
        batch_size = 10  # Process in smaller batches
        embed_list: list[tuple[Any, list[float]]] = []
        
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            batch_texts = [self.text_preprocessor.preprocess(prod.description) for prod in batch]
            
            # Batch API call
            response = self.__openai_client.embeddings.create(
                input=batch_texts,
                model=config.OPENAI_MODEL,
                dimensions=self.__dimension
            )
            
            for j, embedding_data in enumerate(response.data):
                embed_list.append((batch[j].id, embedding_data.embedding))
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)
        
        self.logger.info(f"Generated embeddings for {len(embed_list)} products in batches.")
        return embed_list

    async def _get_embedding_by_prod_id(self, product_id: Any) -> Embedding | None:
        """
        Retrieve an embedding from the database by product ID.
        Args:
            product_id (Any): The product ID.
        Returns:
            Embedding: The embedding object.
        """
        self.logger.debug(f"Fetching embedding for product_id: {product_id}")
        query: Select = embeddings_table.select().where(embeddings_table.c.product_id == product_id)
        record = await db.get_database().fetch_one(query)
        embedding: Embedding | None = Embedding(**dict(record)) if record is not None else None
        if embedding:
            self.logger.info(f"Found embedding for product_id: {product_id}")
        else:
            self.logger.info(f"No embedding found for product_id: {product_id}")
        return embedding

    async def save_embeddings(self) -> None:
        """
        Generate and save embeddings for all products in the database.
        """
        try:
            embeddings_generated: list[tuple[Any, list[float]]] = await self.generate_embeddings()  # type: ignore
            saved_count = 0
            for pair in embeddings_generated:
                if await self._get_embedding_by_prod_id(pair[0]):
                    continue
                # Convert list[float] to bytes for storage
                embedding_bytes = np.array(pair[1], dtype=np.float32).tobytes()
                query: Insert = embeddings_table.insert().values(dict(EmbeddingIn(
                    product_id=pair[0], embedding=embedding_bytes)))
                await db.get_database().execute(query)
                saved_count += 1
            self.logger.info(f"Saved new embeddings to the database.")
        except Exception as e:
            self.logger.error(f"Error saving embeddings")
            raise

    async def get_embeddings_from_db(self) -> list[Embedding]:
        """
        Retrieve embeddings with caching for better performance.
        """
        # Check cache first
        if self._embeddings_cache is not None:
            return self._embeddings_cache
        
        # Try Redis cache
        cached_embeddings = await cache_manager.get_cached_embeddings()
        if cached_embeddings:
            self._embeddings_cache = cached_embeddings
            return cached_embeddings
        
        try:
            # Limit initial query for faster startup
            query: Select = embeddings_table.select().limit(100)
            records = await db.get_database().fetch_all(query)
            
            embeddings: list[Embedding] = []
            for record in records:
                try:
                    embedding = Embedding(
                        id=record["id"],
                        product_id=record["product_id"],
                        embedding=np.frombuffer(record["embedding"], dtype=np.float32).tolist(),
                        created_at=record["created_at"]
                    )
                    embeddings.append(embedding)
                except Exception as e:
                    self.logger.warning(f"Skipping invalid embedding record: {e}")
                    continue
            
            # Cache results
            self._embeddings_cache = embeddings
            await cache_manager.set_cached_embeddings(embeddings, ttl=1800)  # 30 min cache
            
            self.logger.info(f"Fetched {len(embeddings)} embeddings from database.")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error fetching embeddings from database: {e}")
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

    @staticmethod
    async def get_popular_products(limit: int = 5) -> list[Product]:
        """Get most popular products based on like count"""

        try:
            query_popular_products: Select = products_table.select().join(
                user_interactions_table,
                products_table.c.id == user_interactions_table.c.product_id
            ).where(user_interactions_table.c.liked == True).group_by(products_table.c.id).order_by(desc(func.count(user_interactions_table.c.product_id))).limit(limit)
                
            result = await db.get_database().fetch_all(query=query_popular_products)
            return [Product(**dict(product)) for product in result]
                
        except Exception as e:
            logging.getLogger("api.helpers.embedding.EmbeddingProcessor").error(f"Error in get_popular_products: {e}")
            # Final fallback: return any products
            return []

    async def get_liked_products_per_user(self, user_id: UUID4)-> list[Product] | Any:
        user_liked_interactions: list[UserProductInteractionInDB] = await self.__like_service.get_user_likes(user_id)

        recommended_products: list[Product | None] = []
        for interaction in user_liked_interactions:
            query: Select = products_table.select().where(products_table.c.id == interaction.product_id)
            result = await db.get_database().fetch_one(query)
            product: Product | None = Product(**dict(result)) if result is not None else None
            recommended_products.append(product)
        return recommended_products if recommended_products else []

    async def get_recommended_products_per_user(self, user: UserWithoutPassword, embeddings_list: list[Embedding])-> list[Product] | Any:
        liked_products: list[Product] | Any = await self.get_liked_products_per_user(user.id)
        if not liked_products:
            return []
        
        data_generated: list[tuple[Any, list[float]]] = await self.generate_embeddings(products=liked_products) # type: ignore
        embeddings: list[list[float]] = [pair[1] for pair in data_generated]
        average: np.ndarray | Any = self.get_embeddings_average(embeddings)
        
        faiss_manager: FaissManager = FaissManager(embeddings_list)
        faiss_manager.update_index()
        products_list: list[Product] = await faiss_manager.search(average)  # Sin k para obtener todos
        return products_list


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

    async def search(self, query: np.ndarray, k: int = None) -> list[Product]:
        """Optimized search with batch product fetching"""
        # Use all available embeddings if k is None
        search_k = k if k is not None else self.index.ntotal
        self.logger.debug(f"Searching FAISS index for top {search_k} results.")
        
        if self.index.ntotal == 0:
            self.logger.warning("FAISS index is empty.")
            return []
            
        if query.ndim == 1:
            query = query.reshape(1, -1).astype(np.float32)
            
        try:
            _, indexes = self.index.search(query, min(search_k, self.index.ntotal)) #type: ignore
            ind_list = indexes[0].tolist()
            
            # Batch fetch products
            product_ids = []
            for idx in ind_list:
                if 0 <= idx < len(self.embed_list):
                    product_ids.append(self.embed_list[idx].product_id)
            
            # Fetch products in batch
            recommended_products = await self._batch_fetch_products(product_ids)
            return recommended_products
            
        except Exception as e:
            self.logger.error(f"Error in FAISS search: {e}")
            return []
    
    async def _batch_fetch_products(self, product_ids: list) -> list[Product]:
        """Fetch multiple products efficiently"""
        products = []
        for product_id in product_ids:
            try:
                product = await product_processor.get_product_by_id(product_id)
                if product:
                    products.append(product)
            except Exception as e:
                self.logger.warning(f"Failed to fetch product {product_id}: {e}")
                continue
        return products