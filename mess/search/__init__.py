"""
Music Similarity Search

FAISS-based similarity search using layer-specific MERT embeddings.

Public API:
    LayerBasedRecommender - Main recommendation engine using validated layer mappings
    SimilaritySearchEngine - Low-level FAISS search interface
    FAISSIndex - FAISS index wrapper
    LayerIndexBuilder - Build per-layer indices
    resolve_aspects - Map musical aspects to layers via probing results
"""

from .recommender import LayerBasedRecommender
from .similarity import SimilaritySearchEngine
from .faiss_index import FAISSIndex
from .indices import LayerIndexBuilder
from .aspects import resolve_aspects, load_discovery_results, ASPECT_REGISTRY

__all__ = [
    'LayerBasedRecommender',
    'SimilaritySearchEngine',
    'FAISSIndex',
    'LayerIndexBuilder',
    'resolve_aspects',
    'load_discovery_results',
    'ASPECT_REGISTRY',
]
