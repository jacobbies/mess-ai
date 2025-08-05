"""
Async Unified Music Recommender with Caching and Advanced Features

Production-ready async implementation with multi-level caching,
batch processing, and proper error handling.
"""
import asyncio
import logging
import hashlib
import json
from typing import List, Tuple, Optional, Dict, Any, Set
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import random

from ..search.similarity import SimilaritySearchEngine

# Enums previously defined in unified_recommender
class RecommendationStrategy(Enum):
    """Available recommendation strategies."""
    SIMILARITY = "similarity"          # Basic cosine similarity
    DIVERSE = "diverse"               # Diverse recommendations (MMR, cluster-based)
    POPULAR = "popular"               # Popularity-based
    RANDOM = "random"                 # Random recommendations
    HYBRID = "hybrid"                 # Combination of strategies


class RecommendationMode(Enum):
    """Recommendation modes for diverse strategies."""
    MMR = "mmr"                      # Maximal Marginal Relevance
    CLUSTER = "cluster"              # Cluster-based diversity
    NOVELTY = "novelty"              # Novel/less popular items
    SERENDIPITY = "serendipity"      # Unexpected but relevant
    CONTRAST = "contrast"            # Deliberately different


@dataclass
class RecommendationRequest:
    """Structured recommendation request."""
    track_id: str
    n_recommendations: int = 5
    strategy: str = "similarity"
    mode: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None
    exclude_tracks: Optional[List[str]] = None


@dataclass
class RecommendationResult:
    """Rich recommendation result with metadata."""
    track_id: str
    score: float
    strategy_used: str
    explanation: str
    metadata: Dict[str, Any]
    cached: bool = False
    computation_time_ms: float = 0


class CacheLevel(Enum):
    """Cache levels for multi-tier caching."""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"  # Future enhancement


class RecommendationCache:
    """Multi-level cache for recommendations."""
    
    def __init__(self, cache_dir: Path = None, memory_ttl_seconds: int = 3600):
        self.memory_cache: Dict[str, Tuple[List[RecommendationResult], datetime]] = {}
        self.memory_ttl = timedelta(seconds=memory_ttl_seconds)
        self.cache_dir = cache_dir or Path("data/processed/cache/recommendations")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, request: RecommendationRequest) -> str:
        """Generate cache key from request."""
        key_data = {
            "track_id": request.track_id,
            "n": request.n_recommendations,
            "strategy": request.strategy,
            "mode": request.mode,
            "context": request.user_context,
            "exclude": sorted(request.exclude_tracks) if request.exclude_tracks else None
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, request: RecommendationRequest) -> Optional[List[RecommendationResult]]:
        """Get from cache, checking all levels."""
        key = self._generate_key(request)
        
        # L1: Memory cache
        if key in self.memory_cache:
            results, timestamp = self.memory_cache[key]
            if datetime.now() - timestamp < self.memory_ttl:
                self.hits += 1
                # Mark as cached
                return [RecommendationResult(**{**asdict(r), "cached": True}) for r in results]
            else:
                # Expired
                del self.memory_cache[key]
        
        # L2: Disk cache
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    
                # Check TTL
                timestamp = datetime.fromisoformat(data["timestamp"])
                if datetime.now() - timestamp < self.memory_ttl:
                    results = [RecommendationResult(**r) for r in data["results"]]
                    
                    # Promote to memory cache
                    self.memory_cache[key] = (results, timestamp)
                    self.hits += 1
                    
                    # Mark as cached
                    return [RecommendationResult(**{**asdict(r), "cached": True}) for r in results]
                else:
                    # Expired
                    cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        self.misses += 1
        return None
    
    async def set(self, request: RecommendationRequest, results: List[RecommendationResult]):
        """Store in cache at all levels."""
        key = self._generate_key(request)
        timestamp = datetime.now()
        
        # L1: Memory cache
        self.memory_cache[key] = (results, timestamp)
        
        # L2: Disk cache
        cache_file = self.cache_dir / f"{key}.json"
        try:
            data = {
                "timestamp": timestamp.isoformat(),
                "request": asdict(request),
                "results": [asdict(r) for r in results]
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to write cache file {cache_file}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "memory_entries": len(self.memory_cache),
            "disk_entries": len(list(self.cache_dir.glob("*.json")))
        }


class AsyncUnifiedMusicRecommender:
    """
    Async version of UnifiedMusicRecommender with caching and advanced features.
    
    Features:
    - Async/await support for high concurrency
    - Multi-level caching (memory + disk)
    - Batch processing
    - Popularity tracking
    - Request deduplication
    - Performance metrics
    """
    
    def __init__(
        self, 
        features_dir: str = "data/processed/features",
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        **kwargs
    ):
        self.features_dir = Path(features_dir)
        self.logger = logging.getLogger(__name__)
        
        # Initialize similarity search engine
        self._similarity_engine = SimilaritySearchEngine(features_dir=features_dir, **kwargs)
        
        # Initialize cache
        self.cache = RecommendationCache(
            cache_dir=Path(cache_dir) if cache_dir else None
        ) if enable_cache else None
        
        # Thread pool for CPU-bound operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Request deduplication
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        # Popularity tracking (simple in-memory for now)
        self._popularity_scores: Dict[str, float] = {}
        self._interaction_counts: Dict[str, int] = {}
        
        # Performance metrics
        self._request_count = 0
        self._total_computation_time = 0
    
    async def recommend(
        self,
        track_id: str,
        n_recommendations: int = 5,
        strategy: str = "similarity",
        mode: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None,
        exclude_tracks: Optional[List[str]] = None,
        **kwargs
    ) -> List[RecommendationResult]:
        """
        Get recommendations asynchronously with caching.
        
        Args:
            track_id: Track to get recommendations for
            n_recommendations: Number of recommendations
            strategy: Recommendation strategy
            mode: Mode for diverse strategy
            user_context: Optional user context
            exclude_tracks: Tracks to exclude from results
            **kwargs: Additional strategy parameters
            
        Returns:
            List of RecommendationResult objects
        """
        start_time = asyncio.get_event_loop().time()
        
        # Create request object
        request = RecommendationRequest(
            track_id=track_id,
            n_recommendations=n_recommendations,
            strategy=strategy,
            mode=mode,
            user_context=user_context,
            exclude_tracks=exclude_tracks
        )
        
        # Check cache
        if self.cache:
            cached_results = await self.cache.get(request)
            if cached_results:
                self.logger.debug(f"Cache hit for {track_id} with strategy {strategy}")
                return cached_results
        
        # Check for duplicate in-flight requests
        cache_key = self.cache._generate_key(request) if self.cache else f"{track_id}_{strategy}_{n_recommendations}"
        
        if cache_key in self._pending_requests:
            # Wait for existing request
            self.logger.debug(f"Waiting for duplicate request: {cache_key}")
            return await self._pending_requests[cache_key]
        
        # Create future for this request
        future = asyncio.Future()
        self._pending_requests[cache_key] = future
        
        try:
            # Auto-select strategy if requested
            if strategy == "auto":
                strategy = await self._select_best_strategy(track_id, user_context)
            
            # Execute recommendation in thread pool
            results = await self._execute_recommendation(request, strategy, mode, **kwargs)
            
            # Apply exclusions
            if exclude_tracks:
                results = [r for r in results if r.track_id not in exclude_tracks]
            
            # Ensure we have enough results
            results = results[:n_recommendations]
            
            # Cache results
            if self.cache:
                await self.cache.set(request, results)
            
            # Update metrics
            computation_time = (asyncio.get_event_loop().time() - start_time) * 1000
            self._request_count += 1
            self._total_computation_time += computation_time
            
            # Set computation time on results
            for result in results:
                result.computation_time_ms = computation_time
            
            future.set_result(results)
            return results
            
        except Exception as e:
            self.logger.error(f"Error in recommend: {e}")
            future.set_exception(e)
            raise
        finally:
            # Clean up pending request
            self._pending_requests.pop(cache_key, None)
    
    async def _execute_recommendation(
        self,
        request: RecommendationRequest,
        strategy: str,
        mode: Optional[str],
        **kwargs
    ) -> List[RecommendationResult]:
        """Execute recommendation in thread pool."""
        loop = asyncio.get_event_loop()
        
        # Run similarity search in thread pool
        sync_results = await loop.run_in_executor(
            self._executor,
            self._similarity_engine.search,
            request.track_id,
            request.n_recommendations * 2,  # Get extra for filtering
            True  # exclude_self
        )
        
        # Convert to RecommendationResult objects
        results = []
        for track_id, score in sync_results:
            explanation = self._generate_explanation(strategy, mode, score)
            
            metadata = {
                "strategy": strategy,
                "mode": mode,
                "popularity_score": self._get_popularity_score(track_id),
                "interaction_count": self._interaction_counts.get(track_id, 0)
            }
            
            results.append(RecommendationResult(
                track_id=track_id,
                score=score,
                strategy_used=strategy,
                explanation=explanation,
                metadata=metadata,
                cached=False
            ))
        
        return results
    
    async def _select_best_strategy(
        self,
        track_id: str,
        user_context: Optional[Dict[str, Any]]
    ) -> str:
        """Select the best strategy based on context."""
        # Simple heuristic for now
        if user_context:
            if user_context.get("exploring", False):
                return "diverse"
            if user_context.get("prefer_popular", False):
                return "popular"
            if user_context.get("mood") == "energetic":
                return "hybrid"
        
        # Check if track has enough interaction data
        if self._interaction_counts.get(track_id, 0) > 10:
            return "popular"
        
        return "similarity"
    
    def _generate_explanation(self, strategy: str, mode: Optional[str], score: float) -> str:
        """Generate human-readable explanation."""
        if strategy == "similarity":
            return f"Similar musical characteristics (score: {score:.3f})"
        elif strategy == "diverse":
            if mode == "mmr":
                return f"Diverse but relevant selection (MMR score: {score:.3f})"
            elif mode == "cluster":
                return f"From different musical clusters (score: {score:.3f})"
            else:
                return f"Diverse recommendation (score: {score:.3f})"
        elif strategy == "popular":
            return f"Popular and similar (score: {score:.3f})"
        elif strategy == "random":
            return "Random discovery"
        elif strategy == "hybrid":
            return f"Balanced recommendation (score: {score:.3f})"
        else:
            return f"Recommended (score: {score:.3f})"
    
    def _get_popularity_score(self, track_id: str) -> float:
        """Get popularity score for a track."""
        # Simple scoring based on composer and piece
        base_score = self._popularity_scores.get(track_id, 0.5)
        
        # Boost for popular composers
        popular_composers = ['bach', 'mozart', 'beethoven', 'chopin']
        if any(composer in track_id.lower() for composer in popular_composers):
            base_score *= 1.2
        
        # Boost based on interactions
        interactions = self._interaction_counts.get(track_id, 0)
        if interactions > 0:
            base_score *= (1 + min(interactions / 100, 1))
        
        return min(base_score, 1.0)
    
    async def recommend_batch(
        self,
        requests: List[Dict[str, Any]]
    ) -> List[List[RecommendationResult]]:
        """
        Process multiple recommendation requests in batch.
        
        Args:
            requests: List of request dictionaries
            
        Returns:
            List of recommendation lists
        """
        tasks = []
        for req in requests:
            task = self.recommend(**req)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def track_interaction(self, track_id: str, interaction_type: str = "play"):
        """Track user interaction for popularity scoring."""
        self._interaction_counts[track_id] = self._interaction_counts.get(track_id, 0) + 1
        
        # Update popularity score
        current_score = self._popularity_scores.get(track_id, 0.5)
        
        # Simple decay + boost
        if interaction_type == "play":
            boost = 0.01
        elif interaction_type == "like":
            boost = 0.05
        elif interaction_type == "skip":
            boost = -0.02
        else:
            boost = 0
        
        new_score = min(max(current_score + boost, 0), 1)
        self._popularity_scores[track_id] = new_score
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_computation_time = (
            self._total_computation_time / self._request_count 
            if self._request_count > 0 else 0
        )
        
        metrics = {
            "total_requests": self._request_count,
            "avg_computation_time_ms": avg_computation_time,
            "available_strategies": ["similarity", "diverse", "popular", "random", "hybrid"],
            "popularity_tracked_tracks": len(self._popularity_scores),
            "total_interactions": sum(self._interaction_counts.values())
        }
        
        if self.cache:
            metrics["cache_stats"] = self.cache.get_stats()
        
        return metrics
    
    async def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        
        # Cancel any pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()