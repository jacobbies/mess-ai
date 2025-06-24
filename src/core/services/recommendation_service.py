"""
Recommendation service for music similarity search.
Handles all recommendation-related business logic.
"""
from typing import Dict, List, Tuple, Optional
import logging

from mess_ai.models.metadata import TrackMetadata

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for music recommendation and similarity search."""
    
    def __init__(self, recommender, metadata_dict: Dict[str, TrackMetadata]):
        """Initialize with recommender engine and metadata."""
        self.recommender = recommender
        self.metadata_dict = metadata_dict
        logger.info(f"RecommendationService initialized with {len(metadata_dict)} tracks")
    
    def get_recommendations(
        self, 
        track_name: str, 
        top_k: int = 5,
        strategy: str = "similar"
    ) -> Dict:
        """
        Get music recommendations based on MERT feature similarity.
        
        Args:
            track_name: Name of the reference track
            top_k: Number of recommendations to return
            strategy: Recommendation strategy to use
            
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
        
        # Get similar tracks based on strategy
        if hasattr(self.recommender, 'get_recommendations'):
            # Use diverse recommender if available
            similar_tracks = self.recommender.get_recommendations(
                clean_track_name, strategy=strategy, top_k=top_k
            )
        else:
            # Fallback to standard recommender
            similar_tracks = self.recommender.find_similar_tracks(clean_track_name, top_k=top_k)
        
        # Get metadata for reference track (strip -SMD suffix for lookup)
        ref_metadata_key = clean_track_name.replace('-SMD', '')
        ref_metadata = self.metadata_dict.get(ref_metadata_key)
        
        # Format response with metadata
        recommendations = []
        
        for track, score in similar_tracks:
            # Strip -SMD suffix for metadata lookup
            metadata_key = track.replace('-SMD', '')
            track_metadata = self.metadata_dict.get(metadata_key)
            
            rec_data = self._format_recommendation(track, score, track_metadata)
            recommendations.append(rec_data)
        
        response = {
            "reference_track": clean_track_name,
            "recommendations": recommendations,
            "strategy": strategy,
            "total_tracks": len(self.recommender.get_track_names())
        }
        
        # Add reference track metadata if available
        if ref_metadata:
            response["reference_metadata"] = self._format_track_metadata(ref_metadata)
        
        return response
    
    def get_similar_tracks_raw(
        self, 
        track_name: str, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get raw similarity results without metadata formatting.
        
        Args:
            track_name: Name of the reference track
            top_k: Number of similar tracks to return
            
        Returns:
            List of (track_name, similarity_score) tuples
        """
        if self.recommender is None:
            raise ValueError("Recommender not available")
        
        clean_track_name = track_name.replace('.wav', '')
        return self.recommender.find_similar_tracks(clean_track_name, top_k=top_k)
    
    def get_track_names(self) -> List[str]:
        """Get all available track names."""
        if self.recommender is None:
            return []
        return self.recommender.get_track_names()
    
    def is_available(self) -> bool:
        """Check if recommendation service is available."""
        return self.recommender is not None
    
    def get_faiss_status(self) -> Dict:
        """Get FAISS index status information."""
        if not self.recommender:
            return {"available": False, "reason": "Recommender not loaded"}
        
        if not hasattr(self.recommender, 'search_engine'):
            return {"available": False, "reason": "Search engine not available"}
        
        faiss_ready = self.recommender.search_engine.faiss_index is not None
        
        return {
            "available": faiss_ready,
            "index_type": type(self.recommender.search_engine.faiss_index).__name__ if faiss_ready else None,
            "total_vectors": self.recommender.search_engine.faiss_index.index.ntotal if faiss_ready else 0
        }
    
    def _format_recommendation(
        self, 
        track: str, 
        score: float, 
        metadata: Optional[TrackMetadata]
    ) -> Dict:
        """Format a single recommendation with metadata."""
        if metadata:
            return {
                "track_id": track,
                "title": metadata.title,
                "composer": metadata.composer,
                "composer_full": metadata.composer_full,
                "era": metadata.era,
                "form": metadata.form,
                "key_signature": metadata.key_signature,
                "similarity_score": round(score, 4),
                "filename": metadata.filename,
                "tags": metadata.tags
            }
        else:
            # Fallback if metadata not found
            return {
                "track_id": track,
                "title": track.replace('_', ' ').replace('-', ' '),
                "similarity_score": round(score, 4),
                "filename": f"{track}.wav"
            }
    
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
    
    def compare_strategies(self, track_name: str, top_k: int = 5) -> Dict:
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
        
        # Get strategy comparison if available
        if hasattr(self.recommender, 'get_strategy_comparison'):
            comparison = self.recommender.get_strategy_comparison(clean_track_name, top_k)
            
            # Format with metadata
            formatted_comparison = {}
            for strategy, data in comparison.items():
                formatted_recs = []
                for track, score in data['recommendations']:
                    metadata_key = track.replace('-SMD', '')
                    track_metadata = self.metadata_dict.get(metadata_key)
                    rec_data = self._format_recommendation(track, score, track_metadata)
                    formatted_recs.append(rec_data)
                
                formatted_comparison[strategy] = {
                    "recommendations": formatted_recs,
                    "diversity_score": data.get('diversity_score', 0.0),
                    "avg_similarity": data.get('avg_similarity', 0.0)
                }
            
            return {
                "reference_track": clean_track_name,
                "strategies": formatted_comparison
            }
        else:
            # Fallback: just return standard recommendations
            standard_recs = self.get_recommendations(clean_track_name, top_k)
            return {
                "reference_track": clean_track_name,
                "strategies": {
                    "similar": {
                        "recommendations": standard_recs['recommendations'],
                        "diversity_score": 0.0,
                        "avg_similarity": sum(r['similarity_score'] for r in standard_recs['recommendations']) / len(standard_recs['recommendations']) if standard_recs['recommendations'] else 0.0
                    }
                }
            }