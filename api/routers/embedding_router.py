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
from api.services.recommendations.product_service import product_processor
import numpy as np
import asyncio
from typing import Optional, Literal
from fastapi import Query

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
        print(f"😁😁😁😁{len(embeddings_list)}")
        _faiss_manager = FaissManager(embed_list=embeddings_list)
        _faiss_manager.update_index()
    return _faiss_manager

@router.post("/content/chat", 
            response_model=Response, 
            status_code=status.HTTP_200_OK,
            description="Get recommended products based on content description with filters"
            )
async def recommender_by_content(
    payload: Description,
    background_tasks: BackgroundTasks,
    current_user: UserWithoutPassword = Depends(get_current_user),
    skip: int = Query(default=0, ge=0, description="Number of products to skip"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of products to return"),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    sort_by: Optional[Literal["name", "price"]] = Query(default=None, description="Sort by field"),
    order: Optional[Literal["asc", "desc"]] = Query(default="asc", description="Sort order"),
    search: Optional[str] = Query(default=None, description="Additional search term")
):
    text = payload.description.strip()

    if not text or len(text) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Description must be at least 3 characters long."
        )
    
    try:
        embed_processor = await get_embed_processor()
        faiss_manager = await get_faiss_manager()
        
        text_embedded = np.array(await embed_processor.generate_embeddings(text=text))
        
        recommended_products = await faiss_manager.search(text_embedded, k=200)  # Get more for filtering
        
        # Apply filters to recommended products
        filtered_products = []
        for product in recommended_products:
            # Category filter
            if category and hasattr(product, 'category') and product.category and product.category.lower() != category.lower():
                continue
                
            # Additional search filter
            if search:
                search_term = search.lower()
                product_name = product.name.lower() if product.name else ""
                product_desc = product.description.lower() if hasattr(product, 'description') and product.description else ""
                if search_term not in product_name and search_term not in product_desc:
                    continue
                    
            filtered_products.append(product)
        
        # Sort products
        if sort_by:
            reverse = (order == "desc")
            if sort_by == "name":
                filtered_products.sort(key=lambda x: x.name or "", reverse=reverse)
            elif sort_by == "price":
                filtered_products.sort(key=lambda x: x.price or 0, reverse=reverse)
        
        # Apply pagination
        paginated_products = filtered_products[skip:skip + limit]
        
        if not paginated_products:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No products found matching your criteria. Please try again with different filters."
            )
        
        answer_task = asyncio.create_task(
            asyncio.to_thread(build_answer, text, recommended_products)
        )
        
        answer = await answer_task
        if not answer:
            answer = f"Based on your query '{text}', here are some recommended products:"
        
        return Response(
            answer=answer,
            products=paginated_products
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in content recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during recommendation."
        )

@router.get("/interaction", 
            response_model=list[Product], 
            status_code=status.HTTP_200_OK,
            description="Get recommended products based on user interactions with filters"
            )
async def recommender_by_interaction(
    background_tasks: BackgroundTasks,
    current_user: UserWithoutPassword = Depends(get_current_user),
    skip: int = Query(default=0, ge=0, description="Number of products to skip"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of products to return"),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    sort_by: Optional[Literal["name", "price"]] = Query(default=None, description="Sort by field"),
    order: Optional[Literal["asc", "desc"]] = Query(default="asc", description="Sort order"),
    search: Optional[str] = Query(default=None, description="Search term")
):
    try:
        embed_processor = await get_embed_processor()
        embeddings_list = await embed_processor.get_embeddings_from_db()
        
        
        if not embeddings_list:
            return []
        
        liked_products = await product_processor.get_liked_products_per_user(current_user.id)
        logger.info(f"Found {len(liked_products)} liked products for user {current_user.id}")
        
        if not liked_products:
            logger.info("No liked products found, returning empty list")
            return []
        
        liked_embeddings: list[list[float]] = []
        for product in liked_products:
            try:
                embedding = await embed_processor.get_embedding_by_prod_id(product.id)
                liked_embeddings.append(embedding.embedding)
                logger.info(f"Added embedding for product {product.id}")
            except Exception as e:
                logger.warning(f"Could not get embedding for product {product.id}: {e}")
                continue
        
        logger.info(f"Found {len(liked_embeddings)} embeddings for liked products")
        
        if not liked_embeddings:
            logger.info("No embeddings found for liked products")
            return []
        
        average_embedding = embed_processor.get_embeddings_average(liked_embeddings)
        
        faiss_manager = await get_faiss_manager()
        recommended_products = await faiss_manager.search(average_embedding, k=200)  # Get more for filtering
        
        # Apply filters to recommended products
        filtered_products = []
        for product in recommended_products:
            # Skip products already liked by user
            if any(liked.id == product.id for liked in liked_products):
                continue
                
            # Category filter
            if category and hasattr(product, 'category') and product.category and product.category.lower() != category.lower():
                continue
                
            # Search filter - search in name and description
            if search:
                search_term = search.lower()
                product_name = product.name.lower() if product.name else ""
                product_desc = product.description.lower() if hasattr(product, 'description') and product.description else ""
                if search_term not in product_name and search_term not in product_desc:
                    continue
                    
            filtered_products.append(product)
        
        # Sort products
        if sort_by:
            reverse = (order == "desc")
            if sort_by == "name":
                filtered_products.sort(key=lambda x: x.name or "", reverse=reverse)
            elif sort_by == "price":
                filtered_products.sort(key=lambda x: x.price or 0, reverse=reverse)
        
        # Apply pagination
        paginated_products = filtered_products[skip:skip + limit]
        
        logger.info(f"Recommended products generated successfully: {len(paginated_products)} products")
        return paginated_products
        
    except Exception as e:
        logger.error(f"Error in interaction recommendation")
        return []

@router.get("/popular", 
            response_model=list[Product], 
            status_code=status.HTTP_200_OK,
            description="Get most popular products based on like count with filters"
            )
async def get_popular_products(
    skip: int = Query(default=0, ge=0, description="Number of products to skip"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of products to return"),
    category: Optional[str] = Query(default=None, description="Filter by category"),
    sort_by: Optional[Literal["name", "price", "popularity"]] = Query(default="popularity", description="Sort by field"),
    order: Optional[Literal["asc", "desc"]] = Query(default="desc", description="Sort order"),
    search: Optional[str] = Query(default=None, description="Search term")
):
    try:
        embed_processor = await get_embed_processor()
        popular_products = await product_processor.get_popular_products(limit=200)  # Get more for filtering
        
        # Apply filters
        filtered_products = []
        for product in popular_products:
            # Category filter
            if category and hasattr(product, 'category') and product.category and product.category.lower() != category.lower():
                continue
                
            # Search filter
            if search:
                search_term = search.lower()
                product_name = product.name.lower() if product.name else ""
                product_desc = product.description.lower() if hasattr(product, 'description') and product.description else ""
                if search_term not in product_name and search_term not in product_desc:
                    continue
                    
            filtered_products.append(product)
        
        # Sort products (popularity is already sorted from the query)
        if sort_by and sort_by != "popularity":
            reverse = (order == "desc")
            if sort_by == "name":
                filtered_products.sort(key=lambda x: x.name or "", reverse=reverse)
            elif sort_by == "price":
                filtered_products.sort(key=lambda x: x.price or 0, reverse=reverse)
        elif sort_by == "popularity" and order == "asc":
            filtered_products.reverse()  # Reverse the default desc order
        
        # Apply pagination
        paginated_products = filtered_products[skip:skip + limit]
        
        return paginated_products
        
    except Exception as e:
        logger.error(f"Error getting popular products: {e}")
        return []
