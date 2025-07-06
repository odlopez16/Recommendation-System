from uuid import UUID
from fastapi import APIRouter, status, Depends
from api.services.auth_service import get_current_user
from api.models.auth_model import UserInDB
from api.models.products_model import Product
from api.services.product_service import ProductProcessor
import logging

logger = logging.getLogger("api.routers.products_router")

router = APIRouter(tags=["products"])

@router.get("/products", response_model=list[Product], status_code=status.HTTP_200_OK)
async def get_products(current_user: UserInDB = Depends(get_current_user))-> list[Product]:
    return await ProductProcessor().get_products()

@router.get("/{product_id}", response_model=Product, status_code=status.HTTP_200_OK)
async def get_product_by_id(product_id: UUID, current_user: UserInDB = Depends(get_current_user))-> Product:
    return await ProductProcessor().get_product_by_id(product_id)