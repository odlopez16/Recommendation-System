import sys
from pathlib import Path
from venv import logger
sys.path.append(str(Path(__file__).resolve().parent.parent))
import faiss # type: ignore
from typing import Any
from sqlalchemy import Insert, Select
from api.models.embedding_model import Embedding, EmbeddingIn
from api.models.products_model import Product
from api.schemas.embedding_schema import embeddings_table
from api.schemas.product_schema import products_table
from api.database.database_config import primary_database as embed_db
from api.database.database_config import secondary_database as prod_db
import numpy as np
from openai import OpenAI
from api.services.preprocessing_service import TextPreprocessor
from api.services.product_service import ProductProcessor
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

    async def _get_products_from_db(self) -> list[Product]:
        """
        Retrieve all products from the database.
        Returns:
            list[Product]: List of Product objects.
        """
        return await ProductProcessor().get_products()

    async def generate_embeddings(self, text: str = "") -> list[float] | list[tuple[Any, list[float]]]:
        """
        Generate embeddings for a given text or for all products if text is empty.
        Args:
            text (str): The input text. If empty, generate for all products.
        Returns:
            list[float] | list[tuple[Any, list[float]]]: Embedding(s) generated.
        """
        text_preprocessor = TextPreprocessor()
        try:
            if text == "":
                products: list[Product] = await self._get_products_from_db()
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

    async def _get_embedding_by_prod_id(self, product_id: Any) -> Embedding:
        """
        Retrieve an embedding from the database by product ID.
        Args:
            product_id (Any): The product ID.
        Returns:
            Embedding: The embedding object.
        """
        self.logger.debug(f"Fetching embedding for product_id: {product_id}")
        query: Select = embeddings_table.select().where(embeddings_table.c.product_id == product_id)
        embedding: Embedding = await embed_db.get_database().fetch_one(query)  # type: ignore
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
                await embed_db.get_database().execute(query)
                saved_count += 1
            self.logger.info(f"Saved {saved_count} new embeddings to the database.")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")
            raise

    async def get_embeddings_from_db(self) -> list[Embedding]:
        """
        Retrieve all embeddings from the database and convert them to list[float].
        Returns:
            list[Embedding]: List of Embedding objects.
        """
        import numpy as np
        try:
            query: Select = embeddings_table.select()
            records = await embed_db.get_database().fetch_all(query)
            embeddings: list[Embedding] = []
            for record in records:
                # Convert bytes back to list[float]
                embedding: list[float] = np.frombuffer(record["embedding"], dtype=np.float32).tolist()
                embeddings.append(Embedding(
                    id=record["id"],
                    product_id=record["product_id"],
                    embedding=embedding
                ))
            self.logger.info(f"Fetched {len(embeddings)} embeddings from the database.")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error fetching embeddings from database: {e}")
            raise



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
        self.logger.info(f"Initialized FAISS index with dimension {dimension}.")

    def update_index(self, ) -> None:
        """
        Add new embeddings to the FAISS index.
        Args:
            embeddings (np.ndarray): Embeddings to add.
        """
        if not self.index.is_trained:
            self.logger.debug("Training FAISS index.")
            self.index.train(self.embeddings.astype(np.float32))  # type: ignore
        self.index.add(self.embeddings.astype(np.float32))  # type: ignore
        self.logger.info(f"Added {self.embeddings.shape[0]} embeddings to FAISS index.")

    async def _get_product_by_id(self, prod_id: Any) -> Product:
        query: Select = products_table.select().where(products_table.c.id == prod_id)
        record = await prod_db.get_database().fetch_one(query)
        if record is None:
            self.logger.error(f"Product with id {prod_id} not found in the database.")
            raise ValueError(f"Product with id {prod_id} not found.")
        self.logger.info(f"Fetched product with id {prod_id} from the database.")
        return Product(**dict(record))

    async def search(self, query: np.ndarray, k: int = 5) -> list[Product]:
        self.logger.debug(f"Searching FAISS index for top {k} results.")
        if query.ndim == 1:
            query = query.reshape(1, -1).astype(np.float32)
        indexes: np.ndarray = self.index.search(query, k)[1][0]  # type: ignore
        ind_list: list = indexes.tolist()
        logger.info(f"Indexes from FAISS 😎: {ind_list}, Type: {type(ind_list)}")
        recommended_products: list[Product] = []
        for idx in ind_list:
            if idx < 0 or idx >= len(self.embed_list):
                self.logger.error(f"Invalid index from FAISS: {idx}")
                continue
            try:
                product_id = self.embed_list[idx].product_id
                product = await self._get_product_by_id(product_id)
                logger.info(f"🙌Product {idx}:{product}")
                if product:
                    recommended_products.append(product)
                else:
                    self.logger.error(f"Product not found or invalid for index {idx} and product_id     {product_id}")
            except Exception as e:
                self.logger.error(f"Error fetching product for index {idx}: {e}")
                continue
        return recommended_products