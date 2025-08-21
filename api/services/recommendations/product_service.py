from pydantic import UUID4
from sqlalchemy import Select, desc, asc, or_
from api.models.products_model import Product
from api.schemas.product_schema import products_table
from api.database.database_config import primary_database as prim_db, secondary_database as sec_db
from fastapi import HTTPException, status
import logging
from typing import Any, Optional, Literal
from sqlalchemy import Insert

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

        query = query.offset(skip).limit(limit)

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

    async def migrate_products(self):
        """
        Migra productos desde la base de datos secundaria a la primaria.
        
        Este método sincroniza los productos entre las bases de datos, copiando
        los productos que existen en la base de datos secundaria pero no en la
        primaria. Evita duplicados verificando la existencia por ID.
        
        Returns:
            None
            
        Raises:
            HTTPException: Si ocurre un error durante la migración
            
        Notas:
            - Los productos existentes en la base primaria no se modifican
            - El proceso es idempotente y seguro para ejecutar múltiples veces
            - Se registra cada operación en los logs para seguimiento
        """
        self.logger.debug("Iniciando migración de productos entre bases de datos.")
        try:
            products: list[Product] = await self.get_products_from_secondary_db()
            migrated_count = 0
            
            for product in products:
                try:
                    # Verificar si el producto ya existe
                    if await self.get_product_by_id(product.id):
                        continue
                        
                    # Convertir el objeto HttpUrl a string si existe
                    product_data = dict(product)
                    if 'image' in product_data and product_data['image'] is not None:
                        product_data['image'] = str(product_data['image'])
                    
                    # Insertar nuevo producto
                    query: Insert = products_table.insert().values(**product_data)
                    await prim_db.get_database().execute(query)
                    migrated_count += 1
                    
                except HTTPException as he:
                    if he.status_code == status.HTTP_404_NOT_FOUND:
                        # Es normal que el producto no exista, continuamos con la inserción
                        # Convertir el objeto HttpUrl a string si existe
                        product_data = dict(product)
                        if 'image' in product_data and product_data['image'] is not None:
                            product_data['image'] = str(product_data['image'])
                            
                        query: Insert = products_table.insert().values(**product_data)
                        await prim_db.get_database().execute(query)
                        migrated_count += 1
                    else:
                        raise

            self.logger.info(f"Migración completada. {migrated_count} productos migrados.")
            
        except Exception as e:
            self.logger.error(f"Error durante la migración de productos: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error durante la migración de productos"
            )

product_processor: ProductProcessor = ProductProcessor()