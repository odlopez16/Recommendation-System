from api.models.embedding_model import Embedding
import logging
from fastapi import APIRouter, HTTPException, Request, status, Depends
from api.services.recommendations.llm_service import build_answer, client
from api.services.recommendations.embedding_service import EmbeddingProcessor, FaissManager
from api.models.products_model import Product
from api.models.others import Description, Response
from api.services.auth.auth_service import get_current_user
from api.models.auth_model import UserInDB, UserWithoutPassword
import numpy as np

logger = logging.getLogger("api.routers.embedding_router")

router = APIRouter(prefix="/recommendation", tags=["embeddings"])

@router.post("/content/chat", 
            response_model=Response, 
            status_code=status.HTTP_200_OK,
            description="Get recommended products based on content description"
            )
async def recommender_by_content(
    payload: Description,
    current_user: UserWithoutPassword = Depends(get_current_user)
):
    text = payload.description

    if not text:
        logger.error("Description cannot be empty.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Description cannot be empty."
        )
    
    embed_processor = EmbeddingProcessor(client)
    text_embedded = np.array(await embed_processor.generate_embeddings(text=text))

    embeddings_list: list[Embedding] = await embed_processor.get_embeddings_from_db()

    if embeddings_list == []:
        await embed_processor.save_embeddings()
        embeddings_list = await embed_processor.get_embeddings_from_db()
        logger.info(f"Retrieved embeddings from the database.😁")
    
    faiss_manager = FaissManager(embed_list=embeddings_list)
    faiss_manager.update_index()

    recommended_products: list[Product] = await faiss_manager.search(text_embedded)
    logger.info(f"Found recommended products.")

    answer: str | None = build_answer(user_query=text, recommended_products=recommended_products)
    if answer is None:
        logger.error("Failed to generate recommendation answer.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendation answer."
        )
    return Response(
        answer=answer,
        products=recommended_products
    )

@router.get("/interaction", 
            response_model=list[Product], 
            status_code=status.HTTP_200_OK,
            description="Get recommended products based on user interactions"
            )
async def recommender_by_interaction(
    current_user: UserWithoutPassword = Depends(get_current_user)
):
    embed_processor: EmbeddingProcessor = EmbeddingProcessor(client)
    embeddings_list: list[Embedding] = await embed_processor.get_embeddings_from_db()

    if embeddings_list == []:
        await embed_processor.save_embeddings()
        embeddings_list = await embed_processor.get_embeddings_from_db()
        logger.info(f"Retrieved embeddings from the database.😁")
    recommended_products: list[Product] = await embed_processor.get_recommended_products_per_user(user=current_user, embeddings_list=embeddings_list)
    return recommended_products if recommended_products else []

@router.get("/popular", 
            response_model=list[Product], 
            status_code=status.HTTP_200_OK,
            description="Get most popular products based on like count"
            )
async def get_popular_products():
    embed_processor: EmbeddingProcessor = EmbeddingProcessor(client)
    popular_products: list[Product] = await embed_processor.get_popular_products()
    return popular_products
