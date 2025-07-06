from api.models.embedding_model import Embedding
import logging
from fastapi import APIRouter, HTTPException, status, Depends
from api.services.llm_service import build_answer, client
from api.services.embedding_service import EmbeddingProcessor, FaissManager
from api.models.products_model import Product
from api.models.others import Description, Response
from api.services.auth_service import get_current_user
from api.models.auth_model import UserInDB
import numpy as np

logger = logging.getLogger("api.routers.embedding_router")

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

@router.post("/recommendation", response_model=Response, status_code=status.HTTP_200_OK)
async def recommender(
    payload: Description,
    current_user: UserInDB = Depends(get_current_user)
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

    await embed_processor.save_embeddings()
    embeddings_list: list[Embedding] = await embed_processor.get_embeddings_from_db()
    logger.info(f"Retrieved {len(embeddings_list)} embeddings from the database.😁")

    if not embeddings_list:
        logger.error("No embeddings found in the database.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No embeddings found"
        )
    
    faiss_manager = FaissManager(embed_list=embeddings_list)
    faiss_manager.update_index()

    recommended_products: list[Product] = await faiss_manager.search(text_embedded)
    logger.info(f"Found {len(recommended_products)} recommended products.Products: {[p.name for p in recommended_products]}")

    answer: str | None = build_answer(user_query=text, recommended_products=recommended_products)
    if answer is None:
        logger.error("Failed to generate recommendation answer.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendation answer."
        )
    return {
            "answer": answer,
            "products": recommended_products
            }