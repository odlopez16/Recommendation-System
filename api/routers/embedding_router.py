from api.models.embedding_model import Embedding
import logging
from fastapi import APIRouter, HTTPException, Request, status, Depends, BackgroundTasks
from api.services.recommendations.llm_service import build_answer, client
from api.services.recommendations.embedding_service import EmbeddingProcessor, FaissManager
from api.models.products_model import Product
from api.models.others import Description, Response
from api.services.auth.auth_service import get_current_user
from api.models.auth_model import UserInDB, UserWithoutPassword
from api.services.cache_service import cache_manager
import numpy as np
import asyncio
from typing import Optional

logger = logging.getLogger("api.routers.embedding_router")

router = APIRouter(prefix="/recommendation", tags=["embeddings"])

# Global instances for better performance
_embed_processor: Optional[EmbeddingProcessor] = None
_faiss_manager: Optional[FaissManager] = None

async def get_embed_processor() -> EmbeddingProcessor:
    """Get or create embedding processor singleton"""
    global _embed_processor
    if _embed_processor is None:
        _embed_processor = EmbeddingProcessor(client)
    return _embed_processor

async def get_faiss_manager() -> FaissManager:
    """Get or create FAISS manager with cached embeddings"""
    global _faiss_manager
    if _faiss_manager is None:
        embed_processor = await get_embed_processor()
        embeddings_list = await embed_processor.get_embeddings_from_db()
        _faiss_manager = FaissManager(embed_list=embeddings_list)
        _faiss_manager.update_index()
    return _faiss_manager

@router.post("/content/chat", 
            response_model=Response, 
            status_code=status.HTTP_200_OK,
            description="Get recommended products based on content description"
            )
async def recommender_by_content(
    payload: Description,
    background_tasks: BackgroundTasks,
    current_user: UserWithoutPassword = Depends(get_current_user)
):
    text = payload.description.strip()

    if not text or len(text) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Description must be at least 3 characters long."
        )
    
    try:
        # Use cached instances
        embed_processor = await get_embed_processor()
        faiss_manager = await get_faiss_manager()
        
        # Generate embedding for user query
        text_embedded = np.array(await embed_processor.generate_embeddings(text=text))
        
        # Search for similar products
        recommended_products = await faiss_manager.search(text_embedded)
        
        if not recommended_products:
            # Background task to refresh embeddings if empty
            background_tasks.add_task(refresh_embeddings_background)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No products found. Please try again later."
            )
        
        # Generate answer asynchronously
        answer_task = asyncio.create_task(
            asyncio.to_thread(build_answer, text, recommended_products)
        )
        
        answer = await answer_task
        if not answer:
            answer = f"Based on your query '{text}', here are some recommended products:"
        
        return Response(
            answer=answer,
            products=recommended_products
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in content recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during recommendation."
        )

async def refresh_embeddings_background():
    """Background task to refresh embeddings"""
    try:
        embed_processor = await get_embed_processor()
        await embed_processor.save_embeddings()
        logger.info("Background embeddings refresh completed")
    except Exception as e:
        logger.error(f"Background embeddings refresh failed: {e}")

@router.get("/interaction", 
            response_model=list[Product], 
            status_code=status.HTTP_200_OK,
            description="Get recommended products based on user interactions"
            )
async def recommender_by_interaction(
    current_user: UserWithoutPassword = Depends(get_current_user)
):
    try:
        embed_processor = await get_embed_processor()
        embeddings_list = await embed_processor.get_embeddings_from_db()
        
        if not embeddings_list:
            return []  # Return empty instead of generating embeddings synchronously
        
        recommended_products = await embed_processor.get_recommended_products_per_user(
            user=current_user, 
            embeddings_list=embeddings_list
        )
        logger.info(f"😎😎😎😎Recommended products: {recommended_products}")
        return recommended_products
        
    except Exception as e:
        logger.error(f"Error in interaction recommendation: {e}")
        return []  # Graceful degradation

@router.get("/popular", 
            response_model=list[Product], 
            status_code=status.HTTP_200_OK,
            description="Get most popular products based on like count"
            )
async def get_popular_products():
    try:
        embed_processor = await get_embed_processor()
        popular_products = await embed_processor.get_popular_products(limit=10)
        return popular_products
    except Exception as e:
        logger.error(f"Error getting popular products: {e}")
        return []  # Graceful degradation
