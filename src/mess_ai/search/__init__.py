"""
FAISS-based similarity search module for music recommendations.
"""

from .similarity import SimilaritySearchEngine
from .faiss_index import FAISSIndex
from .cache import IndexCache

__all__ = ['SimilaritySearchEngine', 'FAISSIndex', 'IndexCache']