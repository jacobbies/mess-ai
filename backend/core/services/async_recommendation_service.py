"""
Async Recommendation service for music similarity search.
Handles all recommendation-related business logic with async support.
"""
from typing import Dict, List, Optional, Any
import logging

from models.responses import TrackMetadata
# TODO: Import from pipeline service via REST API
# These will be replaced with REST API calls to pipeline service
from typing import Any
# Temporary type aliases until pipeline service is connected
RecommendationResult = Dict[str, Any]
RecommendationRequest = Dict[str, Any]

logger = logging.getLogger(__name__)


class AsyncRecommendationService:
    """Async service for music recommendation and similarity search."""
    
    def __init__(self, recommender: Optional[Any], metadata_dict: Dict[str, TrackMetadata]):
        """Initialize with async recommender engine and metadata."""
        self.recommender = recommender
        self.metadata_dict = metadata_dict
        logger.info(f"AsyncRecommendationService initialized with {len(metadata_dict)} tracks")
    
    async def get_recommendations(
        self, 
        track_name: str, 
        top_k: int = 5,
        strategy: str = "similarity",
        mode: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        exclude_tracks: Optional[List[str]] = None
    ) -> Dict:
        """
        Get music recommendations asynchronously.
        
        Args:
            track_name: Name of the reference track
            top_k: Number of recommendations to return
            strategy: Recommendation strategy to use
            mode: Mode for diverse strategy
            user_context: Optional user context
            exclude_tracks: Tracks to exclude from results
            
        Returns:
            Dictionary with recommendations and metadata
            
        Raises:
            ValueError: If track not found or recommender unavailable
        """
        if self.recommender is None:
            raise ValueError("Recommender not available - features not loaded")
        
        # Remove .wav extension if present for consistency
        clean_track_name = track_name.replace('.wav', '')
        
        logger.info(f"Getting {top_k} recommendations for {clean_track_name} using {strategy} strategy")
        
        # Get recommendations using async recommender
        results = await self.recommender.recommend(
            track_id=clean_track_name,
            n_recommendations=top_k,
            strategy=strategy,
            mode=mode,
            user_context=user_context,
            exclude_tracks=exclude_tracks
        )
        
        # Get metadata for reference track (strip -SMD suffix for lookup)
        ref_metadata_key = clean_track_name.replace('-SMD', '')
        ref_metadata = self.metadata_dict.get(ref_metadata_key)
        
        # Format response with metadata
        recommendations = []
        for result in results:
            # Strip -SMD suffix for metadata lookup
            metadata_key = result.track_id.replace('-SMD', '')
            track_metadata = self.metadata_dict.get(metadata_key)
            
            rec_data = self._format_recommendation_from_result(result, track_metadata)
            recommendations.append(rec_data)
        
        response = {
            "reference_track": clean_track_name,
            "recommendations": recommendations,
            "strategy": strategy,
            "mode": mode,
            "cached": any(r.cached for r in results),
            "computation_time_ms": results[0].computation_time_ms if results else 0
        }
        
        # Add reference track metadata if available
        if ref_metadata:
            response["reference_metadata"] = self._format_track_metadata(ref_metadata)
        
        return response
    
    async def get_recommendations_batch(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[Dict]:
        """
        Process multiple recommendation requests in batch.
        
        Args:
            requests: List of request dictionaries
            
        Returns:
            List of recommendation responses
        """
        # Process batch
        batch_results = await self.recommender.recommend_batch(requests)
        
        # Format each result
        responses = []
        for req, results in zip(requests, batch_results):
            track_name = req.get("track_id", "")
            clean_track_name = track_name.replace('.wav', '')
            
            recommendations = []
            for result in results:
                metadata_key = result.track_id.replace('-SMD', '')
                track_metadata = self.metadata_dict.get(metadata_key)
                rec_data = self._format_recommendation_from_result(result, track_metadata)
                recommendations.append(rec_data)
            
            response = {
                "reference_track": clean_track_name,
                "recommendations": recommendations,
                "strategy": req.get("strategy", "similarity"),
                "cached": any(r.cached for r in results),
                "computation_time_ms": results[0].computation_time_ms if results else 0
            }
            
            responses.append(response)
        
        return responses
    
    def track_interaction(self, track_id: str, interaction_type: str = "play"):
        """Track user interaction for popularity scoring."""
        if hasattr(self.recommender, 'track_interaction'):
            self.recommender.track_interaction(track_id, interaction_type)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get recommender performance metrics."""
        if hasattr(self.recommender, 'get_metrics'):
            return self.recommender.get_metrics()
        return {}
    
    def is_available(self) -> bool:
        """Check if recommendation service is available."""
        return self.recommender is not None
    
    def _format_recommendation_from_result(
        self, 
        result: RecommendationResult,
        metadata: Optional[TrackMetadata]
    ) -> Dict:
        """Format a recommendation result with metadata."""
        base_data = {
            "track_id": result.track_id,
            "similarity_score": round(result.score, 4),
            "strategy_used": result.strategy_used,
            "explanation": result.explanation,
            "cached": result.cached,
            "metadata": result.metadata
        }
        
        if metadata:
            base_data.update({
                "title": metadata.title,
                "composer": metadata.composer,
                "composer_full": metadata.composer_full,
                "era": metadata.era,
                "form": metadata.form,
                "key_signature": metadata.key_signature,
                "filename": metadata.filename,
                "tags": metadata.tags
            })
        else:
            # Fallback if metadata not found
            base_data.update({
                "title": result.track_id.replace('_', ' ').replace('-', ' '),
                "filename": f"{result.track_id}.wav"
            })
        
        return base_data
    
    def _format_track_metadata(self, metadata: TrackMetadata) -> Dict:
        """Format track metadata for API response."""
        return {
            "title": metadata.title,
            "composer": metadata.composer,
            "composer_full": metadata.composer_full,
            "era": metadata.era,
            "form": metadata.form,
            "key_signature": metadata.key_signature,
            "tags": metadata.tags
        }
    
    async def compare_strategies(self, track_name: str, top_k: int = 5) -> Dict:
        """
        Compare different recommendation strategies for a given track.
        
        Args:
            track_name: Name of the reference track
            top_k: Number of recommendations per strategy
            
        Returns:
            Dictionary with strategy comparisons
        """
        if self.recommender is None:
            raise ValueError("Recommender not available - features not loaded")
        
        clean_track_name = track_name.replace('.wav', '')
        
        # Define strategies to compare
        strategies_to_compare = [
            {"strategy": "similarity", "name": "Traditional Similarity"},
            {"strategy": "diverse", "mode": "mmr", "name": "Diverse (MMR)"},
            {"strategy": "diverse", "mode": "cluster", "name": "Diverse (Cluster)"},
            {"strategy": "popular", "name": "Popularity-Boosted"},
            {"strategy": "hybrid", "name": "Hybrid Approach"}
        ]
        
        # Get recommendations for each strategy
        comparison_results = {}
        
        for strat_config in strategies_to_compare:
            strategy = strat_config["strategy"]
            mode = strat_config.get("mode")
            name = strat_config["name"]
            
            results = await self.recommender.recommend(
                track_id=clean_track_name,
                n_recommendations=top_k,
                strategy=strategy,
                mode=mode
            )
            
            # Format recommendations
            formatted_recs = []
            for result in results:
                metadata_key = result.track_id.replace('-SMD', '')
                track_metadata = self.metadata_dict.get(metadata_key)
                rec_data = self._format_recommendation_from_result(result, track_metadata)
                formatted_recs.append(rec_data)
            
            # Calculate diversity score (average pairwise distance)
            diversity_score = self._calculate_diversity_score(results)
            avg_similarity = sum(r.score for r in results) / len(results) if results else 0
            
            comparison_results[name] = {
                "recommendations": formatted_recs,
                "diversity_score": round(diversity_score, 4),
                "avg_similarity": round(avg_similarity, 4),
                "cached": any(r.cached for r in results),
                "computation_time_ms": results[0].computation_time_ms if results else 0
            }
        
        return {
            "reference_track": clean_track_name,
            "strategies": comparison_results
        }
    
    def _calculate_diversity_score(self, results: List[RecommendationResult]) -> float:
        """Calculate diversity score for a set of recommendations."""
        if len(results) < 2:
            return 0.0
        
        # Simple diversity metric: 1 - average similarity score
        # (Lower similarity scores indicate more diverse recommendations)
        avg_score = sum(r.score for r in results) / len(results)
        return 1 - avg_score