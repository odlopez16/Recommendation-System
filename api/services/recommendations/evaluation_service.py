"""
Evaluation Service for Recommendation System

This service integrates with the existing system to provide
real-time evaluation capabilities using actual user data.
"""

from typing import Any, List, Dict, Set, Optional, Tuple
import logging
from pydantic import UUID4
from sqlalchemy import Select, func

from api.models.user_interaction_model import UserProductInteractionInDB
from api.models.products_model import Product
from api.schemas.user_interaction_schema import user_interactions_table
from api.schemas.product_schema import products_table
from api.database.database_config import primary_database as db
from api.services.recommendations.evaluation_metrics import EvaluationRunner, RecommendationMetrics
from api.services.recommendations.embedding_service import EmbeddingProcessor
from api.services.recommendations.llm_service import client
from api.services.recommendations.like_service import like_service
from api.services.recommendations.product_service import product_processor

logger = logging.getLogger(__name__)


class RecommendationEvaluationService:
    """
    Service to evaluate recommendation system performance using real data
    """
    
    def __init__(self):
        self.evaluator = EvaluationRunner()
        self.embedding_processor = EmbeddingProcessor(client)
    
    async def get_user_interactions(self, user_id: UUID4) -> Set[UUID4]:
        """Get products that user has liked (positive interactions)"""
        try:
            interactions: list[UserProductInteractionInDB] = await like_service.get_user_likes(user_id=user_id)
            return {interaction.product_id for interaction in interactions}
        except Exception as e:
            logger.error(f"Error getting user interactions")
            return set()
    
    async def get_all_users_with_interactions(self) -> List[UUID4]:
        """Get all users who have at least one interaction"""
        try:
            query: Select = user_interactions_table.select().where(
                user_interactions_table.c.liked == True
            ).distinct(user_interactions_table.c.user_id)
            records = await db.get_database().fetch_all(query)
            return [record["user_id"] for record in records]
        except Exception as e:
            logger.error(f"Error getting users with interactions")
            return []
    
    async def get_product_features(self, product_id: UUID4) -> Set[str]:
        """Extract features from product for diversity calculation"""
        try:
            product: Product = await product_processor.get_product_by_id(product_id)
            if not product:
                return set()
            
            features = set()
            
            # Add category as feature
            if product.category:
                features.add(f"category_{product.category.lower()}")
            
            # Add price range as feature
            if product.price:
                if product.price < 50:
                    features.add("price_low")
                elif product.price < 200:
                    features.add("price_medium")
                else:
                    features.add("price_high")
            
            # Add keywords from name
            if product.name:
                words = product.name.lower().split()
                features.update(f"name_{word}" for word in words[:3])
            
            return features
        except Exception as e:
            logger.error(f"Error getting product features")
            return set()
    
    async def get_product_popularity(self) -> Dict[str, int]:
        """Get popularity count for each product based on interactions"""
        try:
            query: Select = user_interactions_table.select().where(
                user_interactions_table.c.liked == True
            ).group_by(user_interactions_table.c.product_id).with_only_columns(
                user_interactions_table.c.product_id,
                func.count().label('interaction_count')
            )
            records = await db.get_database().fetch_all(query)
            return {record["product_id"]: record["interaction_count"] for record in records}
        except Exception as e:
            logger.error(f"Error getting product popularity")
            return {}
    
    async def get_total_catalog_items(self) -> Set[str]:
        """Get all product IDs in the catalog"""
        try:
            query: Select = products_table.select().with_only_columns(products_table.c.id)
            records = await db.get_database().fetch_all(query)
            return {record["id"] for record in records}
        except Exception as e:
            logger.error(f"Error getting catalog items")
            return set()
    
    async def get_total_users_count(self) -> int:
        """Get total number of users in the system"""
        try:
            query: Select = user_interactions_table.select().with_only_columns(
                func.count(user_interactions_table.c.user_id.distinct())
            )
            result = await db.get_database().fetch_one(query)
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting total users count")
            return 0
    
    async def generate_recommendations_for_user(self, user_id: UUID4, k: int = 20) -> List[str]:
        """Generate recommendations for a user using the existing embedding service"""
        try:
            # Get user's liked products
            liked_products = await self.get_user_interactions(user_id)
            
            if not liked_products:
                # For users without interactions, return popular products
                popularity = await self.get_product_popularity()
                sorted_products = sorted(popularity.items(), key=lambda x: x[1], reverse=True)
                return [product_id for product_id, _ in sorted_products[:k]]
            
            # Get product descriptions for liked products in batch
            query: Select = products_table.select().where(
                products_table.c.id.in_(liked_products)
            )
            records = await db.get_database().fetch_all(query)
            liked_product_descriptions = [record["description"] for record in records if record["description"]]
            
            if not liked_product_descriptions:
                return []
            
            # Create a combined query from liked products
            combined_query = " ".join(liked_product_descriptions[:3])  # Use first 3 descriptions
            
            # Get recommendations using embedding service
            recommendations = await self.embedding_service.get_recommendations(
                query=combined_query,
                limit=k + len(liked_products)  # Get extra to filter out already liked
            )
            
            # Filter out already liked products
            recommended_ids = []
            for rec in recommendations:
                if rec.get('id') not in liked_products and len(recommended_ids) < k:
                    recommended_ids.append(rec.get('id'))
            
            return recommended_ids
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user")
            return []
    
    async def evaluate_system_performance(self, 
                                        sample_users: Optional[List[str]] = None,
                                        k_values: List[int] = [5, 10, 20],
                                        include_diversity: bool = True) -> Dict[str, float]:
        """
        Evaluate the recommendation system performance using real data
        
        Args:
            sample_users: List of user IDs to evaluate (if None, uses all users)
            k_values: List of k values for ranking metrics
            include_diversity: Whether to include diversity metrics
            
        Returns:
            Dictionary with all computed metrics
        """
        try:
            # Get users to evaluate
            if sample_users is None:
                all_users = await self.get_all_users_with_interactions()
                # Sample up to 100 users for performance
                sample_users = all_users[:100] if len(all_users) > 100 else all_users
            
            if not sample_users:
                logger.warning("No users with interactions found for evaluation")
                return {}
            
            logger.info(f"Evaluating system performance for {len(sample_users)} users")
            
            # Generate recommendations and collect relevant items for each user
            user_recommendations = {}
            user_relevant_items = {}
            
            for user_id in sample_users:
                # Generate recommendations
                recommendations = await self.generate_recommendations_for_user(user_id, max(k_values))
                user_recommendations[user_id] = recommendations
                
                # Get relevant items (liked products)
                relevant_items = await self.get_user_interactions(user_id)
                user_relevant_items[user_id] = relevant_items
            
            # Evaluate ranking metrics
            results = self.evaluator.evaluate_ranking_metrics(
                user_recommendations, 
                user_relevant_items, 
                k_values
            )
            
            # Add diversity metrics if requested
            if include_diversity:
                try:
                    # Collect item features in batch
                    all_items = set()
                    for recs in user_recommendations.values():
                        all_items.update(recs)
                    
                    item_features = {}
                    for item_id in all_items:
                        item_features[item_id] = await self.get_product_features(item_id)
                    
                    # Get additional data for diversity metrics
                    item_popularity = await self.get_product_popularity()
                    total_catalog_items = await self.get_total_catalog_items()
                    total_users = await self.get_total_users_count()
                    
                    diversity_results = self.evaluator.evaluate_diversity_metrics(
                        user_recommendations,
                        item_features,
                        item_popularity,
                        total_catalog_items,
                        total_users
                    )
                    
                    results.update(diversity_results)
                    
                except Exception as e:
                    logger.warning(f"Could not compute diversity metrics: {str(e)}")
            
            logger.info(f"Evaluation completed. Results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in system performance evaluation: {str(e)}")
            return {}
    
    async def evaluate_query_recommendations(self, 
                                     query: str, 
                                     expected_relevant_items: Set[str],
                                     k_values: List[int] | None = [5, 10, 20]) -> Dict[str, float] | Any:
        """
        Evaluate recommendations for a specific query
        
        Args:
            query: Search query
            expected_relevant_items: Set of item IDs that should be relevant
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary with ranking metrics
        """
        try:
            # Get recommendations for the query
            recommendations = await self.embedding_service.get_recommendations(query, limit=max(k_values))
            recommended_ids = [rec.get('id') for rec in recommendations if rec.get('id')]
            
            # Calculate metrics
            results = {}
            
            if k_values is not None:
                for k in k_values:
                    precision = RecommendationMetrics.precision_at_k(recommended_ids, expected_relevant_items, k)
                    recall = RecommendationMetrics.recall_at_k(recommended_ids, expected_relevant_items, k)
                    ndcg = RecommendationMetrics.ndcg_at_k(recommended_ids, expected_relevant_items, k)
                    
                    results[f'precision@{k}'] = precision
                    results[f'recall@{k}'] = recall
                    results[f'ndcg@{k}'] = ndcg
                
                return results
            return
            
        except Exception as e:
            logger.error(f"Error evaluating query recommendations")
            return {}
    
    def get_evaluation_summary(self, metrics: Dict[str, float]) -> str:
        """Generate a human-readable summary of evaluation results"""
        if not metrics:
            return "No evaluation metrics available"
        
        summary_parts = []
        
        # Ranking metrics summary
        if 'precision@5' in metrics:
            p5 = metrics['precision@5']
            r5 = metrics['recall@5']
            ndcg5 = metrics['ndcg@5']
            summary_parts.append(f"Ranking@5: P={p5:.3f}, R={r5:.3f}, NDCG={ndcg5:.3f}")
        
        if 'map' in metrics:
            summary_parts.append(f"MAP={metrics['map']:.3f}")
        
        # Diversity metrics summary
        if 'catalog_coverage' in metrics:
            coverage = metrics['catalog_coverage']
            summary_parts.append(f"Coverage={coverage:.3f}")
        
        if 'avg_intra_list_diversity' in metrics:
            diversity = metrics['avg_intra_list_diversity']
            summary_parts.append(f"Diversity={diversity:.3f}")
        
        return " | ".join(summary_parts) if summary_parts else "Evaluation completed"