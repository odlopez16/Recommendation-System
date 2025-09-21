"""
Evaluation Router for Recommendation System Metrics

This router provides endpoints to evaluate the recommendation system
using the metrics implemented in evaluation_metrics.py
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Set, Optional
from pydantic import BaseModel
import logging
import random

from api.services.recommendations.evaluation_metrics import EvaluationRunner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class UserRecommendationData(BaseModel):
    user_id: str
    recommended_items: List[str]
    relevant_items: List[str]


class EvaluationRequest(BaseModel):
    user_data: List[UserRecommendationData]
    k_values: Optional[List[int]] = [5, 10, 20]


class RatingPredictionData(BaseModel):
    predicted_ratings: List[float]
    actual_ratings: List[float]


class DiversityEvaluationRequest(BaseModel):
    user_data: List[UserRecommendationData]
    item_features: Optional[Dict[str, List[str]]] = None
    item_popularity: Optional[Dict[str, int]] = None
    total_catalog_size: Optional[int] = None
    total_users: Optional[int] = None


class EvaluationResponse(BaseModel):
    metrics: Dict[str, float]
    summary: str


@router.post("/ranking-metrics", response_model=EvaluationResponse)
async def evaluate_ranking_metrics(request: EvaluationRequest):
    """
    Evaluate ranking metrics: Precision@k, Recall@k, NDCG@k, MAP
    
    Based on the metrics described in Chapter 1, Section 1.1.3
    """
    try:
        # Convert request data to required format using dict comprehensions
        user_recommendations = {user_data.user_id: user_data.recommended_items for user_data in request.user_data}
        user_relevant_items = {user_data.user_id: set(user_data.relevant_items) for user_data in request.user_data}
        
        # Run evaluation
        evaluator = EvaluationRunner()
        metrics = evaluator.evaluate_ranking_metrics(
            user_recommendations, 
            user_relevant_items, 
            request.k_values
        )
        
        # Generate summary
        summary_parts = []
        for k in request.k_values:
            precision = metrics.get(f'precision@{k}', 0.0)
            recall = metrics.get(f'recall@{k}', 0.0)
            ndcg = metrics.get(f'ndcg@{k}', 0.0)
            summary_parts.append(f"@{k}: P={precision:.3f}, R={recall:.3f}, NDCG={ndcg:.3f}")
        
        map_score = metrics.get('map', 0.0)
        summary = f"Ranking Metrics - {', '.join(summary_parts)}, MAP={map_score:.3f}"
        
        logger.info("Ranking evaluation completed")
        
        return EvaluationResponse(metrics=metrics, summary=summary)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation failed"
        )


@router.post("/prediction-metrics", response_model=EvaluationResponse)
async def evaluate_prediction_metrics(request: RatingPredictionData):
    """
    Evaluate prediction metrics: RMSE, MAE
    
    Based on the metrics described in Chapter 1, Section 1.1.3
    """
    try:
        from api.services.recommendations.evaluation_metrics import RecommendationMetrics
        
        metrics = {}
        
        # Calculate RMSE and MAE
        metrics['rmse'] = RecommendationMetrics.rmse(
            request.predicted_ratings, 
            request.actual_ratings
        )
        metrics['mae'] = RecommendationMetrics.mae(
            request.predicted_ratings, 
            request.actual_ratings
        )
        
        summary = f"Prediction Metrics - RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}"
        
        logger.info("Prediction evaluation completed")
        
        return EvaluationResponse(metrics=metrics, summary=summary)
        
    except Exception as e:
        logger.error("Error in prediction metrics evaluation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation failed"
        )

@router.post("/diversity-metrics", response_model=EvaluationResponse)
async def evaluate_diversity_metrics(request: DiversityEvaluationRequest):
    """
    Evaluate diversity and coverage metrics
    
    Based on the metrics described in Chapter 1, Section 1.1.3
    """
    try:
        # Convert request data to required format
        user_recommendations = {user_data.user_id: user_data.recommended_items for user_data in request.user_data}
        
        # Convert item features to sets if provided
        item_features = {}
        if request.item_features:
            item_features = {k: set(v) for k, v in request.item_features.items()}
        
        # Get catalog items from recommendations if not provided
        total_catalog_items = set()
        for recs in user_recommendations.values():
            total_catalog_items.update(recs)
        
        if request.total_catalog_size:
            # Extend catalog with dummy items if size specified
            dummy_items = {f"item_{i}" for i in range(len(total_catalog_items), request.total_catalog_size)}
            total_catalog_items.update(dummy_items)
        
        total_users = request.total_users or len(user_recommendations)
        
        # Run evaluation
        evaluator = EvaluationRunner()
        metrics = evaluator.evaluate_diversity_metrics(
            user_recommendations,
            item_features,
            request.item_popularity or {},
            total_catalog_items,
            total_users
        )
        
        # Generate summary
        coverage = metrics.get('catalog_coverage', 0.0)
        diversity = metrics.get('avg_intra_list_diversity', 0.0)
        novelty = metrics.get('avg_novelty', 0.0)
        
        summary = f"Diversity Metrics - Coverage={coverage:.3f}, Diversity={diversity:.3f}, Novelty={novelty:.3f}"
        
        logger.info(f"Diversity evaluation completed for {len(request.user_data)} users")
        
        return EvaluationResponse(metrics=metrics, summary=summary)
        
    except Exception as e:
        logger.error("Error in diversity metrics evaluation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation failed"
        )


@router.post("/comprehensive-evaluation", response_model=EvaluationResponse)
async def comprehensive_evaluation(request: DiversityEvaluationRequest):
    """
    Run comprehensive evaluation with all available metrics
    Combines ranking, prediction, and diversity metrics as described in Chapter 1
    """
    try:
        # Convert request data using dict comprehensions
        user_recommendations = {user_data.user_id: user_data.recommended_items for user_data in request.user_data}
        user_relevant_items = {user_data.user_id: set(user_data.relevant_items) for user_data in request.user_data}
        
        # Convert item features to sets if provided
        item_features = None
        if request.item_features:
            item_features = {k: set(v) for k, v in request.item_features.items()}
        
        # Get catalog items
        total_catalog_items = set()
        for recs in user_recommendations.values():
            total_catalog_items.update(recs)
        
        if request.total_catalog_size:
            dummy_items = {f"item_{i}" for i in range(len(total_catalog_items), request.total_catalog_size)}
            total_catalog_items.update(dummy_items)
        
        total_users = request.total_users or len(user_recommendations)
        
        # Run comprehensive evaluation
        evaluator = EvaluationRunner()
        metrics = evaluator.comprehensive_evaluation(
            user_recommendations,
            user_relevant_items,
            item_features,
            request.item_popularity,
            total_catalog_items,
            total_users
        )
        
        # Generate comprehensive summary
        precision_5 = metrics.get('precision@5', 0.0)
        recall_5 = metrics.get('recall@5', 0.0)
        ndcg_5 = metrics.get('ndcg@5', 0.0)
        map_score = metrics.get('map', 0.0)
        coverage = metrics.get('catalog_coverage', 0.0)
        
        summary = (f"Comprehensive Evaluation - "
                  f"P@5={precision_5:.3f}, R@5={recall_5:.3f}, "
                  f"NDCG@5={ndcg_5:.3f}, MAP={map_score:.3f}, "
                  f"Coverage={coverage:.3f}")
        
        logger.info(f"Comprehensive evaluation completed for {len(request.user_data)} users")
        
        return EvaluationResponse(metrics=metrics, summary=summary)
        
    except Exception as e:
        logger.error("Error in comprehensive evaluation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation failed"
        )


@router.post("/real-time-evaluation", response_model=EvaluationResponse)
async def real_time_evaluation(
    sample_size: Optional[int] = 50,
    k_values: Optional[List[int]] = [5, 10, 20],
    include_diversity: bool = True
):
    """
    Evaluate system performance using real user data and interactions
    
    This endpoint uses actual user interactions and generates recommendations
    to evaluate the system's current performance.
    """
    try:
        from api.services.recommendations.evaluation_service import RecommendationEvaluationService
        
        eval_service = RecommendationEvaluationService()
        
        # Get sample of users with interactions
        all_users = await eval_service.get_all_users_with_interactions()
        if not all_users:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No users with interactions found for evaluation"
            )
        
        # Limit sample size for performance
        sample_users = random.sample(all_users, min(sample_size, len(all_users)))
        
        # Run evaluation
        metrics = await eval_service.evaluate_system_performance(
            sample_users=sample_users,
            k_values=k_values,
            include_diversity=include_diversity
        )
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Evaluation failed to produce results"
            )
        
        # Generate summary
        summary = eval_service.get_evaluation_summary(metrics)
        
        logger.info(f"Real-time evaluation completed for {len(sample_users)} users")
        
        return EvaluationResponse(
            metrics=metrics, 
            summary=f"Real-time evaluation ({len(sample_users)} users): {summary}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in real-time evaluation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Real-time evaluation failed"
        )


@router.post("/query-evaluation", response_model=EvaluationResponse)
async def evaluate_query(
    query: str,
    relevant_items: List[str],
    k_values: Optional[List[int]] = [5, 10, 20]
):
    """
    Evaluate recommendations for a specific query against expected relevant items
    
    Useful for testing specific queries and validating recommendation quality
    """
    try:
        from api.services.recommendations.evaluation_service import RecommendationEvaluationService
        
        eval_service = RecommendationEvaluationService()
        
        # Convert to set for evaluation
        expected_relevant = set(relevant_items)
        
        # Evaluate query
        metrics = await eval_service.evaluate_query_recommendations(
            query=query,
            expected_relevant_items=expected_relevant,
            k_values=k_values
        )
        
        if not metrics:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Query evaluation failed"
            )
        
        # Generate summary
        precision_5 = metrics.get('precision@5', 0.0)
        recall_5 = metrics.get('recall@5', 0.0)
        ndcg_5 = metrics.get('ndcg@5', 0.0)
        
        summary = f"Query '{query}': P@5={precision_5:.3f}, R@5={recall_5:.3f}, NDCG@5={ndcg_5:.3f}"
        
        logger.info("Query evaluation completed")
        
        return EvaluationResponse(metrics=metrics, summary=summary)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in query evaluation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query evaluation failed"
        )


@router.get("/metrics-info")
async def get_metrics_info():
    """
    Get information about available evaluation metrics
    """
    return {
        "ranking_metrics": {
            "precision@k": "Proportion of relevant items among k recommended items",
            "recall@k": "Proportion of relevant items recommended among total relevant items",
            "ndcg@k": "Normalized Discounted Cumulative Gain - ranking quality with position discount",
            "map": "Mean Average Precision across all users"
        },
        "prediction_metrics": {
            "rmse": "Root Mean Square Error for rating prediction (lower is better)",
            "mae": "Mean Absolute Error for rating prediction (lower is better)"
        },
        "diversity_metrics": {
            "catalog_coverage": "Proportion of catalog items appearing in recommendations",
            "intra_list_diversity": "Average dissimilarity between items in recommendation lists",
            "novelty": "Average novelty based on item popularity (higher is more novel)"
        },
        "references": [
            "Gunawardana & Shani (2015) - Evaluating recommender systems",
            "Ferrari Dacrema et al. (2019) - Neural recommendation approaches analysis",
            "Kaminskas & Bridge (2016) - Beyond-accuracy objectives survey",
            "Kunaver & Požrl (2017) - Diversity in recommender systems survey"
        ]
    }