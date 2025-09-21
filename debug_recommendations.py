import asyncio
import numpy as np
from api.services.recommendations.embedding_service import EmbeddingProcessor, FaissManager
from api.services.recommendations.llm_service import client
from api.services.recommendations.preprocessing_service import TextPreprocessor
import logging

logger = logging.getLogger("debug_recommendations")

async def debug_content_recommendation(query_text: str):
    """
    Debug function to analyze why content recommendations are not relevant
    """
    print(f"🔍 Debugging query: '{query_text}'")
    
    # Initialize components
    embed_processor = EmbeddingProcessor(client)
    preprocessor = TextPreprocessor()
    
    # 1. Check text preprocessing
    processed_text = preprocessor.preprocess(query_text)
    print(f"📝 Original text: '{query_text}'")
    print(f"📝 Processed text: '{processed_text}'")
    
    # 2. Generate embedding for query
    query_embedding = np.array(await embed_processor.generate_embeddings(text=query_text))
    print(f"🔢 Query embedding shape: {query_embedding.shape}")
    
    # 3. Get all embeddings from database
    all_embeddings = await embed_processor.get_embeddings_from_db()
    print(f"📊 Total embeddings in database: {len(all_embeddings)}")
    
    # 4. Initialize FAISS and search
    faiss_manager = FaissManager(embed_list=all_embeddings)
    faiss_manager.update_index()
    
    # 5. Search with different k values to see results
    for k in [5, 10, 20]:
        recommended_products = await faiss_manager.search(query_embedding, k=k)
        print(f"\n🎯 Top {k} recommendations:")
        
        for i, product in enumerate(recommended_products[:5]):  # Show first 5
            print(f"  {i+1}. {product.name[:50]}...")
            print(f"     Description: {product.description[:100]}...")
            
            # Calculate similarity manually
            product_embedding = await embed_processor.get_embedding_by_prod_id(product.id)
            if product_embedding:
                similarity = np.dot(query_embedding, product_embedding.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(product_embedding.embedding)
                )
                print(f"     Similarity: {similarity:.4f}")
            print()
    
    return recommended_products

# Usage example:
# await debug_content_recommendation("laptop gaming")