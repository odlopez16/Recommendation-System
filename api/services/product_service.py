from sqlalchemy import Select
from api.models.products_model import Product
from api.schemas.product_schema import products_table
from api.database.database_config import secondary_database as prod_db
from fastapi import HTTPException, status
import logging


logger = logging.getLogger("api.services.product_service")


class ProductProcessor:
    def __init__(self):
        self.logger = logging.getLogger("api.helpers.embedding.EmbeddingProcessor")
        
    async def get_products(self) -> list[Product]:
        self.logger.debug("Fetching all products from the database.")
        query: Select = products_table.select()
        records = await prod_db.get_database().fetch_all(query=query)
        products: list[Product] = [Product(**dict(record)) for record in records]
        self.logger.info(f"Fetched {len(products)} products from the database.")
        return products

    async def get_product_by_id(self, product_id: int)-> Product:
        self.logger.debug("Fetching product by id from the database.")
        query: Select = products_table.select().where(products_table.c.id == product_id)
        record = await prod_db.get_database().fetch_one(query)
        product: Product = Product(**dict(record)) # type: ignore
        if not product:
            self.logger.error(f"Product with id {product_id} not found in the database.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Product with id {product_id} not found.")
        self.logger.info(f"Fetched product with id {product_id} from the database.")
        return product

    