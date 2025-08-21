import sys
from pathlib import Path

from pydantic import UUID4
from api.models.user_interaction_model import UserProductInteractionInDB
sys.path.append(str(Path(__file__).resolve().parent.parent))
import faiss #type: ignore
from typing import Any
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
from config import config
import logging


class EmbeddingProcessor:
    """
    Handles the generation, storage, and retrieval of embeddings for products.
    """
    def __init__(self, openai_client: OpenAI, dimension: int = 768) -> None:
        self.logger = logging.getLogger("api.helpers.embedding.EmbeddingProcessor")
        """
        Initialize the EmbeddingProcessor.
        Args:
            openai_client (OpenAI): The OpenAI client for embedding generation.
            dimension (int): The dimension of the embeddings.
        """
        self.__openai_client = openai_client
        self.__dimension = dimension
        self.__like_service = like_service

    async def generate_embeddings(self, text: str = "", products: list[Product] = []) -> list[tuple[Any, list[float]]] | Any:
        """
        Generate embeddings for a given text or for all products if text is empty.
        Args:
            text (str): The input text. If empty, generate for all products.
            products (list[Product]): List of products to generate embeddings for.
        Returns:
            list[float] | list[tuple[Any, list[float]]]: Embedding(s) generated.
        """
        text_preprocessor = TextPreprocessor()
        try:
            if text == "":
                products = await product_processor.get_products_from_primary_db()
                embed_list: list[tuple[Any, list[float]]] = []
                for prod in products:
                    embedding: list[float] = self.__openai_client.embeddings.create(
                        input=text_preprocessor.preprocess(prod.description),
                        model=config.OPENAI_MODEL,
                        dimensions=self.__dimension
                    ).data[0].embedding
                    embed_list.append((prod.id, embedding))
                self.logger.info(f"Generated embeddings for {len(embed_list)} products.")
                return embed_list
            else:
                embed: list[float] = self.__openai_client.embeddings.create(
                    input=text_preprocessor.preprocess(text),
                    model=config.OPENAI_MODEL,
                    dimensions=self.__dimension
                ).data[0].embedding
                self.logger.info("Generated embedding for input text.")
                return embed
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

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

    async def get_embeddings_from_db(self) -> list[Embedding] | Any:
        """
        Retrieve all embeddings from the database and convert them to list[float].
        Returns:
            list[Embedding]: List of Embedding objects.
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
            
            self.logger.info(f"Fetched embeddings from the database.")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error fetching embeddings from database")
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
        products_list: list[Product] = await faiss_manager.search(average)
        return products_list


class FaissManager:
    """
    Manages a FAISS index for similarity search on embeddings.
    """
    def __init__(self, embed_list: list[Embedding], dimension: int = 768):
        self.logger = logging.getLogger("api.helpers.embedding.FaissManager")
        """
        Initialize the FAISS index.
        Args:
            dimension (int): The dimension of the embeddings.
        """
        self.embed_list = embed_list
        self.embeddings: np.ndarray = np.array([e.embedding for e in embed_list])
        self.index = faiss.IndexFlatL2(dimension)
        self.logger.info(f"Initialized FAISS index.")

    def update_index(self) -> None:
        """
        Add new embeddings to the FAISS index.
        """
        if not self.index.is_trained:
            self.logger.debug("Training FAISS index.")
            self.index.train(self.embeddings.astype(np.float32))  # type: ignore
        self.index.add(self.embeddings.astype(np.float32))  # type: ignore
        self.logger.info(f"Added embeddings to FAISS index.")

    async def search(self, query: np.ndarray, k: int = 5) -> list[Product]:
        self.logger.debug(f"Searching FAISS index for top {k} results.")
        if query.ndim == 1:
            query = query.reshape(1, -1).astype(np.float32)
        indexes: np.ndarray = self.index.search(query, k)[1][0]  # type: ignore
        ind_list: list = indexes.tolist()
        recommended_products: list[Product] = []
        for idx in ind_list:
            if idx < 0 or idx >= len(self.embed_list):
                self.logger.error(f"Invalid index from FAISS")
                continue
            try:
                product_id = self.embed_list[idx].product_id
                product: Product = await product_processor.get_product_by_id(product_id)
                if product:
                    recommended_products.append(product)
                else:
                    self.logger.error(f"Product not found or invalid for index")
            except Exception as e:
                self.logger.error(f"Error fetching product")
                continue
        return recommended_products