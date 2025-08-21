from typing import Optional
from databases import Database
from fastapi import HTTPException, status
from pydantic import UUID4
from sqlalchemy import Insert, Select, Update
from api.database.database_config import primary_database as db
from api.models.user_interaction_model import (
    UserProductInteractionCreate,
    UserProductInteractionUpdate,
    UserProductInteractionInDB
)
from api.schemas.user_interaction_schema import user_interactions_table
from api.schemas.product_schema import products_table
from api.models.products_model import Product
from api.models.auth_model import UserInDB
from typing import Any
import logging

logger = logging.getLogger("api.services.like_service")

class LikeService:
    def __init__(self, db: Database):
        self.db = db

    async def set_like_status(
        self,
        user: UserInDB,
        interaction_data: UserProductInteractionCreate
    ) -> UserProductInteractionInDB:
        """
        Set like status for a product by a user.
        If the interaction doesn't exist, it will be created.
        If it exists, it will be updated.
        """
        logger.info(f"👤User {user.id} setting like status for product {interaction_data.product_id}")
        # Check if the product exists
        query_product: Select = products_table.select().where(
            products_table.c.id == interaction_data.product_id
        )
        result = await self.db.fetch_one(query=query_product)
        product_exists: Product | None = Product(**dict(result)) if result is not None else None
        if not product_exists:
            raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Product not found"
                )
            
            # Check if interaction already exists
        query_user_interaction: Select = user_interactions_table.select().where(
            user_interactions_table.c.user_id == user.id,
            user_interactions_table.c.product_id == interaction_data.product_id
            )
        result = await self.db.fetch_one(query=query_user_interaction)
        existing_interaction: UserProductInteractionInDB | None = UserProductInteractionInDB(**dict(result)) if result is not None else None
            
        if existing_interaction:
            # Update existing interaction
            query_update_interaction: Update = user_interactions_table.update().where(
                user_interactions_table.c.user_id == user.id, 
                user_interactions_table.c.product_id == interaction_data.product_id
            ).values(liked=interaction_data.liked)
            await self.db.execute(query_update_interaction)
            return existing_interaction
        else:
            # Create new interaction
            query_insert_interaction: Insert = user_interactions_table.insert().values(user_id=user.id, **dict(interaction_data))
            last_record_id: UUID4 = await self.db.execute(query_insert_interaction)
            result = await self.db.fetch_one(query=user_interactions_table.select().where(user_interactions_table.c.id == last_record_id))
            interaction: UserProductInteractionInDB | None = UserProductInteractionInDB(**dict(result)) if result is not None else None
            
            try:    
                if interaction is None:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to update like status"
                    )
                    
                return UserProductInteractionInDB(**dict(interaction))
                
            except Exception as e:
                logger.error(f"Error setting like status")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="An error occurred while updating like status"
                )

    async def get_user_likes(
        self,
        user_id: UUID4,
        offset: int = 0
    ) -> list[UserProductInteractionInDB]:
        """Get all products liked by a user"""
        query_user_likes: Select = user_interactions_table.select().where(
            user_interactions_table.c.user_id == user_id,
            user_interactions_table.c.liked == True
        ).offset(offset)
        interactions = await self.db.fetch_all(query=query_user_likes)
        
        return [UserProductInteractionInDB(**dict(interaction)) for interaction in interactions]
            

    async def get_product_likes_count(
        self,
        product_id: UUID4
    ) -> int:
        """Get the number of likes for a product"""
        query_product_likes_count: Select = user_interactions_table.select().where(
            user_interactions_table.c.product_id == product_id, 
            user_interactions_table.c.liked == True
        )
        result = await self.db.fetch_all(query=query_product_likes_count)
        return len(result)

like_service = LikeService(db=db.get_database())