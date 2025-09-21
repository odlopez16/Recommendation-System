from pydantic import UUID4
from sqlalchemy import Select, desc, asc, Insert, func
from api.models.embedding_model import Embedding
from api.models.products_model import Product
from api.models.user_interaction_model import UserProductInteractionInDB
from api.schemas.product_schema import products_table
from api.database.database_config import primary_database as prim_db, secondary_database as sec_db
from fastapi import HTTPException, status
import logging
from typing import Optional, Literal, Any
from api.services.recommendations.embedding_service import EmbeddingProcessor, FaissManager
from api.services.recommendations.llm_service import client
from api.models.auth_model import UserWithoutPassword
from api.services.recommendations.like_service import like_service
from api.schemas.user_interaction_schema import user_interactions_table
import numpy as np


logger = logging.getLogger("api.services.product_service")


class ProductProcessor:
    """
    Procesador principal para la gestión de productos en el sistema.
    
    Esta clase maneja todas las operaciones relacionadas con productos,
    incluyendo búsqueda, filtrado, ordenamiento y sincronización entre
    bases de datos primaria y secundaria.
    
    Attributes:
        logger: Logger configurado para el procesador de productos
    """
    
    def __init__(self):
        """
        Inicializa el procesador de productos.
        Configura el logger específico para el seguimiento de operaciones.
        """
        self.__like_service = like_service
        self.logger = logging.getLogger("api.helpers.embedding.EmbeddingProcessor")
        
    async def get_products_from_secondary_db(self) -> list[Product]:
        """
        Obtiene todos los productos de la base de datos secundaria.
        
        Este método se utiliza principalmente para la sincronización inicial
        y la migración de datos entre bases de datos.
        
        Returns:
            list[Product]: Lista de todos los productos en la base de datos secundaria
            
        Raises:
            HTTPException: Si ocurre un error al acceder a la base de datos
        """
        self.logger.debug("Fetching all products from the database.")
        query: Select = products_table.select()
        records = await sec_db.get_database().fetch_all(query=query)
        products: list[Product] = [Product(**dict(record)) for record in records]
        self.logger.info(f"Fetched {len(products)} products from the database.")
        return products

    async def get_products_from_primary_db(
        self, 
        skip: int = 0, 
        limit: int = 100, 
        category: Optional[str] = None,
        sort_by: Optional[Literal["name", "price"]] = None,
        order: Optional[Literal["asc", "desc"]] = "asc",
        search: Optional[str] = None
    ) -> list[Product]:
        """
        Obtiene productos de la base de datos primaria con opciones avanzadas de filtrado.
        
        Este método proporciona una interfaz flexible para buscar y filtrar productos,
        permitiendo paginación, ordenamiento y búsqueda por diferentes criterios.
        
        Args:
            skip (int): Número de productos a saltar para la paginación
            limit (int): Número máximo de productos a retornar por página
            category (str, opcional): Filtrar productos por categoría específica
            sort_by (str, opcional): Campo por el cual ordenar ('name' o 'price')
            order (str, opcional): Dirección del ordenamiento ('asc' o 'desc')
            search (str, opcional): Término de búsqueda para filtrar por nombre
            
        Returns:
            list[Product]: Lista de productos que cumplen con los criterios especificados
            
        Raises:
            HTTPException: Si ocurre un error en la consulta o no se encuentran productos
            ValueError: Si los parámetros de ordenamiento o filtrado son inválidos
            
        Ejemplo:
            >>> await get_products_from_primary_db(
            ...     skip=0,
            ...     limit=10,
            ...     category="electronics",
            ...     sort_by="price",
            ...     order="desc"
            ... )
        """
        self.logger.debug("Fetching products with filters and sorting.")
        query: Select = products_table.select()

        if category:
            query = query.where(products_table.c.category == category)

        if search:
            search_term = f"%{search}%"
            query = query.where(products_table.c.name.ilike(search_term))

        if sort_by:
            sort_column = getattr(products_table.c, sort_by)
            if order == "desc":
                query = query.order_by(desc(sort_column))
            else:
                query = query.order_by(asc(sort_column))

        query = query.offset(skip)

        records = await prim_db.get_database().fetch_all(query=query)
        products: list[Product] = [Product(**dict(record)) for record in records]
        self.logger.info(f"Fetched products from the database.")
        return products if products else []

    async def get_product_by_id(self, product_id: UUID4)-> Product:
        """
        Busca y retorna un producto específico por su ID.
        
        Este método realiza una búsqueda precisa por ID en la base de datos
        primaria y retorna los detalles completos del producto si se encuentra.
        
        Args:
            product_id (UUID4): Identificador único del producto a buscar
            
        Returns:
            Product: Objeto con los detalles del producto encontrado
            
        Raises:
            HTTPException(404): Si el producto no existe en la base de datos
            HTTPException(500): Si ocurre un error interno durante la búsqueda
            
        Ejemplo:
            >>> producto = await get_product_by_id("123e4567-e89b-12d3-a456-426614174000")
        """
        self.logger.debug(f"Fetching product by id from the database.")
        query: Select = products_table.select().where(products_table.c.id == product_id)
        record = await prim_db.get_database().fetch_one(query)
        
        if not record:
            self.logger.error(f"Product not found in the database.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No se encontró el producto"
            )
            
        product: Product = Product(**dict(record))
        self.logger.info(f"Successfully fetched product")
        return product

    async def get_recommended_products_per_user(self, user: UserWithoutPassword, faiss_manager: FaissManager, embed_processor: EmbeddingProcessor)-> list[Product] | Any:
        
        liked_products: list[Product] | Any = await self.get_liked_products_per_user(user.id)
        if not liked_products:
            return []
        
        embeddings: list[list[float]] = []
        for p in liked_products:
            embedding = await embed_processor.get_embedding_by_prod_id(product_id=p.id)
            if embedding is not None:
                embeddings.append(embedding.embedding)
        average: np.ndarray | Any = embed_processor.get_embeddings_average(embeddings)
        
        products_list: list[Product] = await faiss_manager.search(average)  # Sin k para obtener todos
        return products_list
    
    async def batch_fetch_products(self, product_ids: list) -> list[Product]:
        """Fetch multiple products efficiently"""
    
        products: list[Product] = [await self.get_product_by_id(product_id)
                    for product_id in product_ids
                    ]
        return products
    
    async def get_popular_products(self, limit: int = 100) -> list[Product]:
        """Get most popular products based on like count"""

        try:
            query_popular_products: Select = products_table.select().join(
                user_interactions_table,
                products_table.c.id == user_interactions_table.c.product_id
            ).where(user_interactions_table.c.liked == True).group_by(products_table.c.id).order_by(desc(func.count(user_interactions_table.c.product_id))).limit(limit)
                
            result = await prim_db.get_database().fetch_all(query=query_popular_products)
            return [Product(**dict(product)) for product in result]
                
        except Exception as e:
            logging.getLogger("api.helpers.embedding.EmbeddingProcessor").error(f"Error in get_popular_products: {e}")
            # Final fallback: return any products
            return []
    
    async def migrate_products(self):
        """
        Migra productos desde la base de datos secundaria a la primaria.
        Genera embeddings automáticamente para productos nuevos.
        """
        self.logger.debug("Iniciando migración de productos entre bases de datos.")
        try:
            # Get products from secondary DB
            secondary_products: list[Product] = await self.get_products_from_secondary_db()
            self.logger.info(f"Found {len(secondary_products)} products in secondary DB")
            
            # Initialize embedding processor
            embed_processor = EmbeddingProcessor(client)
            existing_embeddings: list[Embedding] = await embed_processor.get_embeddings_from_db()
            
            # Create set of product IDs that already have embeddings for faster lookup
            existing_embedding_ids = {emb.product_id for emb in existing_embeddings}
            self.logger.info(f"Found {len(existing_embeddings)} existing embeddings")
            
            migrated_count = 0
            migrated_products: list[Product] = []
            products_needing_embeddings: list[Product] = []
            
            for product in secondary_products:
                try:
                    # Check if product already exists in primary DB
                    try:
                        await self.get_product_by_id(product.id)
                        continue  # Product already exists, skip
                    except HTTPException:
                        pass  # Product doesn't exist, proceed with migration
                        
                    # Migrate product to primary DB
                    product_data = dict(product)
                    if 'image' in product_data and product_data['image'] is not None:
                        product_data['image'] = str(product_data['image'])
                        
                    query: Insert = products_table.insert().values(**product_data)
                    await prim_db.get_database().execute(query)
                    migrated_count += 1
                    migrated_products.append(product)
                    
                    # Check if product needs embedding generation
                    if product.id not in existing_embedding_ids:
                        products_needing_embeddings.append(product)
                    
                    self.logger.debug(f"Migrated product: {product.name}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to migrate product {product.id}: {e}")
                    continue
                
            # Generate embeddings for products that need them
            if products_needing_embeddings:
                self.logger.info(f"Generating embeddings for {len(products_needing_embeddings)} products")
                embed_generated: list[tuple[Any, list[float]]] = await embed_processor.generate_embeddings(products=products_needing_embeddings)
                await embed_processor.save_embeddings(embeddings_generated=embed_generated)
                self.logger.info(f"Generated and saved {len(embed_generated)} new embeddings")

            self.logger.info(f"Migration completed: {migrated_count} products migrated, {len(products_needing_embeddings)} embeddings generated")
            
        except Exception as e:
            self.logger.error(f"Error durante la migración de productos: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error durante la migración de productos"
            )

    async def get_liked_products_per_user(self, user_id: UUID4)-> list[Product] | Any:
        user_liked_interactions: list[UserProductInteractionInDB] = await self.__like_service.get_user_likes(user_id)

        recommended_products: list[Product | None] = []
        for interaction in user_liked_interactions:
            query: Select = products_table.select().where(products_table.c.id == interaction.product_id)
            result = await prim_db.get_database().fetch_one(query)
            product: Product | None = Product(**dict(result)) if result is not None else None
            recommended_products.append(product)
        return recommended_products if recommended_products else []

product_processor: ProductProcessor = ProductProcessor()