from uuid import UUID
from fastapi import APIRouter, Depends, status, HTTPException

from api.services.recommendations.like_service import like_service
from api.models.user_interaction_model import (
    UserProductInteractionCreate,
    UserProductInteractionInDB
)
from api.models.auth_model import UserInDB, UserWithoutPassword
from api.services.auth.auth_service import get_current_user
from api.models.others import NumLikes
import logging

logger = logging.getLogger("api.routers.likes_router")

router = APIRouter(
    prefix="/likes",
    tags=["likes"]
)

@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    description="Like or dislike a product",
    response_model=UserProductInteractionInDB
)
async def set_like_status(
    interaction: UserProductInteractionCreate,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Like or dislike a product.
    
    - **product_id**: UUID of the product to like/dislike
    - **liked**: True to like, False to dislike
    """
    try:
        result: UserProductInteractionInDB = await like_service.set_like_status(current_user, interaction)
        return result
    except Exception as e:
        logger.error(f"Error setting like status 😭{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request"
        )

@router.get(
    "/me",
    status_code=status.HTTP_200_OK,
    description="Get all products liked by the current user",
    response_model=list[UserProductInteractionInDB]
)
async def get_my_likes(
    current_user: UserWithoutPassword = Depends(get_current_user)
):
    """
    Get all products liked by the current user.
    """
    try:
        interactions = await like_service.get_user_likes(
            user_id=current_user.id
        )
        return [UserProductInteractionInDB(**dict(item)) for item in interactions]
    except Exception as e:
        logger.error(f"Error in get_my_likes")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching your liked products"
        )

@router.get(
    "/count/{product_id}",
    status_code=status.HTTP_200_OK,
    description="Get the number of likes for a specific product",
    response_model=NumLikes
)
async def get_product_likes_count(
    product_id: UUID,
    current_user: UserWithoutPassword = Depends(get_current_user)
):
    """
    Get the number of likes for a specific product.
    
    - **product_id**: UUID of the product
    """
    try:
        response: int = await like_service.get_product_likes_count(product_id)
        return NumLikes(
            num_likes=response
        )
    except Exception as e:
        logger.error(f"Error in get_product_likes_count: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching like count"
        )
