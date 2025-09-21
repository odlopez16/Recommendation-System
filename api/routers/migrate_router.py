from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from api.services.auth.auth_service import get_current_user
from api.models.auth_model import UserWithoutPassword
from api.services.recommendations.product_service import product_processor
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/migrate", tags=["migration"])

@router.get("/products", status_code=status.HTTP_200_OK)
async def migrate_products(
    background_tasks: BackgroundTasks,
    current_user: UserWithoutPassword = Depends(get_current_user)
):
    """Migrate products from secondary to primary database"""
    try:
        background_tasks.add_task(run_migration)
        return {"message": "Product migration started in background"}
    except Exception as e:
        logger.error(f"Error starting migration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start migration"
        )

async def run_migration():
    """Background task to run product migration"""
    try:
        await product_processor.migrate_products()
        logger.info("Product migration completed successfully")
    except Exception as e:
        logger.error(f"Migration failed: {e}")