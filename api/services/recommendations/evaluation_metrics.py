"""
Evaluation Metrics for Recommendation Systems

This module implements the evaluation metrics mentioned in Chapter 1:
- Precision@k, Recall@k, NDCG@k (Ranking metrics)
- RMSE, MAE (Prediction metrics)
- Coverage and Diversity metrics
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import math
from collections import defaultdict


class RecommendationMetrics:
    """
    Implementation of recommendation system evaluation metrics
    based on academic literature from Chapter 1.
    """
    
    @staticmethod
    def precision_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
        """
        Precision@k: Proportion of relevant items among the k recommended items.
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: Set of relevant item IDs for the user
            k: Number of top recommendations to consider
            
        Returns:
            Precision@k score (0.0 to 1.0)
        """
        if k <= 0 or not recommended_items:
            return 0.0
            
        top_k = recommended_items[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
        
        return relevant_in_top_k / min(k, len(top_k))
    
    @staticmethod
    def recall_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
        """
        Recall@k: Proportion of relevant items recommended among total relevant items.
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: Set of relevant item IDs for the user
            k: Number of top recommendations to consider
            
        Returns:
            Recall@k score (0.0 to 1.0)
        """
        if not relevant_items or k <= 0:
            return 0.0
            
        top_k = recommended_items[:k]
        relevant_in_top_k = sum(1 for item in top_k if item in relevant_items)
        
        return relevant_in_top_k / len(relevant_items)
    
    @staticmethod
    def ndcg_at_k(recommended_items: List[str], relevant_items: Set[str], k: int) -> float:
        """
        NDCG@k: Normalized Discounted Cumulative Gain at k.
        Measures ranking quality with position-based discounting.
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: Set of relevant item IDs for the user
            k: Number of top recommendations to consider
            
        Returns:
            NDCG@k score (0.0 to 1.0)
        """
        if k <= 0 or not relevant_items:
            return 0.0
            
        # Calculate DCG@k
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG@k (ideal DCG)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant_items))))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mean_average_precision(user_recommendations: Dict[str, List[str]], 
                             user_relevant_items: Dict[str, Set[str]]) -> float:
        """
        MAP: Mean Average Precision across all users.
        
        Args:
            user_recommendations: Dict mapping user_id to list of recommended items
            user_relevant_items: Dict mapping user_id to set of relevant items
            
        Returns:
            MAP score (0.0 to 1.0)
        """
        if not user_recommendations:
            return 0.0
            
        ap_scores = []
        
        for user_id, recommended in user_recommendations.items():
            relevant = user_relevant_items.get(user_id, set())
            if not relevant:
                continue
                
            # Calculate Average Precision for this user
            relevant_count = 0
            precision_sum = 0.0
            
            for i, item in enumerate(recommended):
                if item in relevant:
                    relevant_count += 1
                    precision_sum += relevant_count / (i + 1)
            
            ap = precision_sum / len(relevant) if relevant else 0.0
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    @staticmethod
    def rmse(predicted_ratings: List[float], actual_ratings: List[float]) -> float:
        """
        RMSE: Root Mean Square Error for rating prediction.
        
        Args:
            predicted_ratings: List of predicted ratings
            actual_ratings: List of actual ratings
            
        Returns:
            RMSE score (lower is better)
        """
        if len(predicted_ratings) != len(actual_ratings) or not predicted_ratings:
            return float('inf')
            
        mse = np.mean([(pred - actual) ** 2 for pred, actual in zip(predicted_ratings, actual_ratings)])
        return math.sqrt(mse)
    
    @staticmethod
    def mae(predicted_ratings: List[float], actual_ratings: List[float]) -> float:
        """
        MAE: Mean Absolute Error for rating prediction.
        
        Args:
            predicted_ratings: List of predicted ratings
            actual_ratings: List of actual ratings
            
        Returns:
            MAE score (lower is better)
        """
        if len(predicted_ratings) != len(actual_ratings) or not predicted_ratings:
            return float('inf')
            
        return np.mean([abs(pred - actual) for pred, actual in zip(predicted_ratings, actual_ratings)])
    
    @staticmethod
    def catalog_coverage(recommended_items_all_users: List[List[str]], 
                        total_catalog_items: Set[str]) -> float:
        """
        Catalog Coverage: Proportion of catalog items that appear in recommendations.
        
        Args:
            recommended_items_all_users: List of recommendation lists for all users
            total_catalog_items: Set of all items in the catalog
            
        Returns:
            Coverage score (0.0 to 1.0)
        """
        if not total_catalog_items:
            return 0.0
            
        recommended_items = set()
        for user_recs in recommended_items_all_users:
            recommended_items.update(user_recs)
        
        return len(recommended_items) / len(total_catalog_items)
    
    @staticmethod
    def intra_list_diversity(recommended_items: List[str], 
                           item_features: Dict[str, Set[str]]) -> float:
        """
        Intra-list Diversity: Average dissimilarity between items in a recommendation list.
        
        Args:
            recommended_items: List of recommended item IDs
            item_features: Dict mapping item_id to set of features
            
        Returns:
            Diversity score (0.0 to 1.0, higher is more diverse)
        """
        if len(recommended_items) < 2:
            return 0.0
            
        similarities = []
        
        for i in range(len(recommended_items)):
            for j in range(i + 1, len(recommended_items)):
                item1, item2 = recommended_items[i], recommended_items[j]
                
                features1 = item_features.get(item1, set())
                features2 = item_features.get(item2, set())
                
                if not features1 or not features2:
                    similarity = 0.0
                else:
                    # Jaccard similarity
                    intersection = len(features1 & features2)
                    union = len(features1 | features2)
                    similarity = intersection / union if union > 0 else 0.0
                
                similarities.append(similarity)
        
        # Diversity is 1 - average similarity
        return 1.0 - np.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def novelty_score(recommended_items: List[str], 
                     item_popularity: Dict[str, int], 
                     total_users: int) -> float:
        """
        Novelty: Average novelty of recommended items based on popularity.
        
        Args:
            recommended_items: List of recommended item IDs
            item_popularity: Dict mapping item_id to number of interactions
            total_users: Total number of users in the system
            
        Returns:
            Novelty score (higher is more novel)
        """
        if not recommended_items or total_users <= 0:
            return 0.0
            
        novelty_scores = []
        
        for item in recommended_items:
            popularity = item_popularity.get(item, 0)
            # Avoid log(0) by adding small epsilon
            probability = max(popularity / total_users, 1e-10)
            novelty = -math.log2(probability)
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)


class EvaluationRunner:
    """
    Helper class to run comprehensive evaluation of recommendation systems.
    """
    
    def __init__(self):
        self.metrics = RecommendationMetrics()
    
    def evaluate_ranking_metrics(self, 
                                user_recommendations: Dict[str, List[str]], 
                                user_relevant_items: Dict[str, Set[str]], 
                                k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate ranking metrics for all users.
        
        Args:
            user_recommendations: Dict mapping user_id to list of recommended items
            user_relevant_items: Dict mapping user_id to set of relevant items
            k_values: List of k values to evaluate
            
        Returns:
            Dict with metric names and average scores
        """
        results = {}
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            
            for user_id, recommended in user_recommendations.items():
                relevant = user_relevant_items.get(user_id, set())
                if not relevant:
                    continue
                
                precision_scores.append(self.metrics.precision_at_k(recommended, relevant, k))
                recall_scores.append(self.metrics.recall_at_k(recommended, relevant, k))
                ndcg_scores.append(self.metrics.ndcg_at_k(recommended, relevant, k))
            
            results[f'precision@{k}'] = np.mean(precision_scores) if precision_scores else 0.0
            results[f'recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
            results[f'ndcg@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        # Calculate MAP
        results['map'] = self.metrics.mean_average_precision(user_recommendations, user_relevant_items)
        
        return results
    
    def evaluate_diversity_metrics(self, 
                                  user_recommendations: Dict[str, List[str]], 
                                  item_features: Dict[str, Set[str]], 
                                  item_popularity: Dict[str, int], 
                                  total_catalog_items: Set[str], 
                                  total_users: int) -> Dict[str, float]:
        """
        Evaluate diversity and coverage metrics.
        
        Args:
            user_recommendations: Dict mapping user_id to list of recommended items
            item_features: Dict mapping item_id to set of features
            item_popularity: Dict mapping item_id to interaction count
            total_catalog_items: Set of all catalog items
            total_users: Total number of users
            
        Returns:
            Dict with diversity metric scores
        """
        results = {}
        
        # Catalog coverage
        all_recommendations = list(user_recommendations.values())
        results['catalog_coverage'] = self.metrics.catalog_coverage(all_recommendations, total_catalog_items)
        
        # Average intra-list diversity
        diversity_scores = []
        novelty_scores = []
        
        for recommended in user_recommendations.values():
            diversity_scores.append(self.metrics.intra_list_diversity(recommended, item_features))
            novelty_scores.append(self.metrics.novelty_score(recommended, item_popularity, total_users))
        
        results['avg_intra_list_diversity'] = np.mean(diversity_scores) if diversity_scores else 0.0
        results['avg_novelty'] = np.mean(novelty_scores) if novelty_scores else 0.0
        
        return results
    
    def comprehensive_evaluation(self, 
                               user_recommendations: Dict[str, List[str]], 
                               user_relevant_items: Dict[str, Set[str]], 
                               item_features: Optional[Dict[str, Set[str]]] = None, 
                               item_popularity: Optional[Dict[str, int]] = None, 
                               total_catalog_items: Optional[Set[str]] = None, 
                               total_users: Optional[int] = None, 
                               k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Run comprehensive evaluation with all available metrics.
        
        Returns:
            Dict with all computed metrics
        """
        results = {}
        
        # Ranking metrics
        ranking_results = self.evaluate_ranking_metrics(user_recommendations, user_relevant_items, k_values)
        results.update(ranking_results)
        
        # Diversity metrics (if data available)
        if all(x is not None for x in [item_features, item_popularity, total_catalog_items, total_users]):
            diversity_results = self.evaluate_diversity_metrics(
                user_recommendations, item_features, item_popularity, total_catalog_items, total_users
            )
            results.update(diversity_results)
        
        return results