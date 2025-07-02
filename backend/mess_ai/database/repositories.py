"""Repository classes for PostgreSQL database operations using SQLAlchemy."""

import uuid
import numpy as np
import zlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from sqlalchemy import select, and_, or_, func, delete
from sqlalchemy.orm import selectinload

from .client import BaseRepository, postgres_client
from .models import Recording, Embedding, EmbeddingChunk, SimilarityScore

logger = logging.getLogger(__name__)


class RecordingRepository(BaseRepository):
    """Repository for managing recordings in the database."""
    
    async def create_recording(
        self,
        title: str,
        dataset_source: str,
        original_filename: str,
        audio_local_path: Optional[str] = None,
        audio_s3_key: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new recording."""
        async with self.get_session() as session:
            recording = Recording(
                title=title,
                dataset_source=dataset_source,
                original_filename=original_filename,
                audio_local_path=audio_local_path,
                audio_s3_key=audio_s3_key,
                duration_seconds=duration_seconds,
                metadata=metadata or {}
            )
            
            session.add(recording)
            await session.commit()
            await session.refresh(recording)
            
            logger.info(f"Created recording {recording.id}: {title}")
            return str(recording.id)
    
    async def get_recording(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Get a recording by ID."""
        async with self.get_session() as session:
            result = await session.execute(
                select(Recording)
                .where(Recording.id == recording_id)
                .options(selectinload(Recording.embeddings))
            )
            recording = result.scalar_one_or_none()
            
            if recording:
                return {
                    'id': str(recording.id),
                    'title': recording.title,
                    'dataset_source': recording.dataset_source,
                    'original_filename': recording.original_filename,
                    'audio_local_path': recording.audio_local_path,
                    'audio_s3_key': recording.audio_s3_key,
                    'duration_seconds': float(recording.duration_seconds) if recording.duration_seconds else None,
                    'metadata': recording.metadata,
                    'created_at': recording.created_at,
                    'updated_at': recording.updated_at,
                    'embeddings_count': len(recording.embeddings)
                }
            return None
    
    async def list_recordings(
        self,
        dataset_source: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List recordings with optional filtering."""
        async with self.get_session() as session:
            query = select(Recording)
            
            if dataset_source:
                query = query.where(Recording.dataset_source == dataset_source)
            
            query = query.order_by(Recording.created_at.desc()).offset(offset).limit(limit)
            
            result = await session.execute(query)
            recordings = result.scalars().all()
            
            return [
                {
                    'id': str(r.id),
                    'title': r.title,
                    'dataset_source': r.dataset_source,
                    'original_filename': r.original_filename,
                    'duration_seconds': float(r.duration_seconds) if r.duration_seconds else None,
                    'created_at': r.created_at
                }
                for r in recordings
            ]
    
    async def search_recordings(
        self,
        query: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search recordings by title or metadata."""
        async with self.get_session() as session:
            search_pattern = f"%{query}%"
            
            result = await session.execute(
                select(Recording)
                .where(
                    or_(
                        Recording.title.ilike(search_pattern),
                        Recording.metadata.astext.ilike(search_pattern)
                    )
                )
                .order_by(Recording.created_at.desc())
                .limit(limit)
            )
            
            recordings = result.scalars().all()
            
            return [
                {
                    'id': str(r.id),
                    'title': r.title,
                    'dataset_source': r.dataset_source,
                    'original_filename': r.original_filename,
                    'metadata': r.metadata
                }
                for r in recordings
            ]
    
    async def update_recording_metadata(
        self,
        recording_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update recording metadata."""
        async with self.get_session() as session:
            result = await session.execute(
                select(Recording).where(Recording.id == recording_id)
            )
            recording = result.scalar_one_or_none()
            
            if recording:
                # Merge metadata
                if recording.metadata:
                    recording.metadata.update(metadata)
                else:
                    recording.metadata = metadata
                
                await session.commit()
                logger.info(f"Updated metadata for recording {recording_id}")
                return True
            
            return False
    
    async def delete_recording(self, recording_id: str) -> bool:
        """Delete a recording and all related data."""
        async with self.get_session() as session:
            result = await session.execute(
                delete(Recording).where(Recording.id == recording_id)
            )
            await session.commit()
            
            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"Deleted recording {recording_id}")
            
            return deleted


class EmbeddingRepository(BaseRepository):
    """Repository for managing embeddings and chunks."""
    
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    
    async def store_embedding(
        self,
        recording_id: str,
        embedding_vector: np.ndarray,
        model_name: str = "mert-v1-95M",
        model_version: str = "1.0",
        feature_type: str = "aggregated",
        extraction_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store an embedding with automatic chunking for large vectors."""
        async with self.get_session() as session:
            # Create embedding record
            embedding = Embedding(
                recording_id=recording_id,
                model_name=model_name,
                model_version=model_version,
                feature_type=feature_type,
                dimension=embedding_vector.shape[-1] if len(embedding_vector.shape) > 0 else 1,
                extraction_params=extraction_params or {}
            )
            
            session.add(embedding)
            await session.flush()  # Get the embedding ID
            
            # Compress and chunk the embedding
            compressed = zlib.compress(embedding_vector.astype(np.float32).tobytes())
            
            # Store chunks
            chunks = []
            for i in range(0, len(compressed), self.CHUNK_SIZE):
                chunk_data = compressed[i:i + self.CHUNK_SIZE]
                chunk = EmbeddingChunk(
                    embedding_id=embedding.id,
                    chunk_index=i // self.CHUNK_SIZE,
                    chunk_data=chunk_data,
                    chunk_size=len(chunk_data)
                )
                chunks.append(chunk)
            
            session.add_all(chunks)
            await session.commit()
            
            logger.info(f"Stored embedding {embedding.id} with {len(chunks)} chunks")
            return str(embedding.id)
    
    async def get_embedding(
        self,
        recording_id: str,
        model_name: str = "mert-v1-95M",
        feature_type: str = "aggregated"
    ) -> Optional[np.ndarray]:
        """Retrieve and reconstruct an embedding from chunks."""
        async with self.get_session() as session:
            # Get embedding metadata
            result = await session.execute(
                select(Embedding)
                .where(
                    and_(
                        Embedding.recording_id == recording_id,
                        Embedding.model_name == model_name,
                        Embedding.feature_type == feature_type
                    )
                )
                .order_by(Embedding.extraction_timestamp.desc())
            )
            
            embedding = result.scalar_one_or_none()
            if not embedding:
                return None
            
            # Get all chunks
            chunks_result = await session.execute(
                select(EmbeddingChunk)
                .where(EmbeddingChunk.embedding_id == embedding.id)
                .order_by(EmbeddingChunk.chunk_index)
            )
            
            chunks = chunks_result.scalars().all()
            if not chunks:
                return None
            
            # Reconstruct embedding
            full_data = b''.join(chunk.chunk_data for chunk in chunks)
            
            # Decompress and convert to numpy
            decompressed = zlib.decompress(full_data)
            embedding_array = np.frombuffer(decompressed, dtype=np.float32)
            
            # Reshape if needed
            if embedding.dimension > 1:
                total_elements = len(embedding_array)
                if total_elements % embedding.dimension == 0:
                    rows = total_elements // embedding.dimension
                    embedding_array = embedding_array.reshape(rows, embedding.dimension)
            
            return embedding_array
    
    async def list_embeddings(
        self,
        recording_id: str
    ) -> List[Dict[str, Any]]:
        """List all embeddings for a recording."""
        async with self.get_session() as session:
            result = await session.execute(
                select(Embedding)
                .where(Embedding.recording_id == recording_id)
                .order_by(Embedding.extraction_timestamp.desc())
            )
            
            embeddings = result.scalars().all()
            
            return [
                {
                    'id': str(e.id),
                    'model_name': e.model_name,
                    'model_version': e.model_version,
                    'feature_type': e.feature_type,
                    'dimension': e.dimension,
                    'extraction_timestamp': e.extraction_timestamp,
                    'extraction_params': e.extraction_params
                }
                for e in embeddings
            ]
    
    async def delete_embedding(self, embedding_id: str) -> bool:
        """Delete an embedding and its chunks."""
        async with self.get_session() as session:
            result = await session.execute(
                delete(Embedding).where(Embedding.id == embedding_id)
            )
            await session.commit()
            
            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"Deleted embedding {embedding_id}")
            
            return deleted
    
    async def get_all_embeddings_for_similarity(
        self,
        model_name: str = "mert-v1-95M",
        feature_type: str = "aggregated"
    ) -> List[Tuple[str, np.ndarray]]:
        """Get all embeddings for similarity computation."""
        async with self.get_session() as session:
            result = await session.execute(
                select(Embedding.recording_id, Embedding.id)
                .where(
                    and_(
                        Embedding.model_name == model_name,
                        Embedding.feature_type == feature_type
                    )
                )
            )
            
            embeddings_info = result.all()
            
            # Retrieve all embeddings
            embeddings = []
            for recording_id, embedding_id in embeddings_info:
                embedding_vector = await self.get_embedding(str(recording_id), model_name, feature_type)
                if embedding_vector is not None:
                    embeddings.append((str(recording_id), embedding_vector))
            
            return embeddings


class SimilarityRepository(BaseRepository):
    """Repository for managing similarity scores."""
    
    async def store_similarity(
        self,
        recording_id_a: str,
        recording_id_b: str,
        similarity_score: float,
        model_name: str = "mert-v1-95M"
    ) -> None:
        """Store a similarity score between two recordings."""
        # Ensure consistent ordering
        if recording_id_a > recording_id_b:
            recording_id_a, recording_id_b = recording_id_b, recording_id_a
        
        async with self.get_session() as session:
            # Check if similarity already exists
            result = await session.execute(
                select(SimilarityScore)
                .where(
                    and_(
                        SimilarityScore.recording_id_a == recording_id_a,
                        SimilarityScore.recording_id_b == recording_id_b,
                        SimilarityScore.model_name == model_name
                    )
                )
            )
            
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update existing score
                existing.similarity_score = similarity_score
                existing.computed_at = func.now()
            else:
                # Create new score
                similarity = SimilarityScore(
                    recording_id_a=recording_id_a,
                    recording_id_b=recording_id_b,
                    model_name=model_name,
                    similarity_score=similarity_score
                )
                session.add(similarity)
            
            await session.commit()
    
    async def get_similar_recordings(
        self,
        recording_id: str,
        model_name: str = "mert-v1-95M",
        limit: int = 10,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Get recordings similar to the given recording."""
        async with self.get_session() as session:
            # Query similarities where recording_id is either recording_a or recording_b
            result = await session.execute(
                select(
                    SimilarityScore.recording_id_a,
                    SimilarityScore.recording_id_b,
                    SimilarityScore.similarity_score
                )
                .where(
                    and_(
                        or_(
                            SimilarityScore.recording_id_a == recording_id,
                            SimilarityScore.recording_id_b == recording_id
                        ),
                        SimilarityScore.model_name == model_name,
                        SimilarityScore.similarity_score >= min_score
                    )
                )
                .order_by(SimilarityScore.similarity_score.desc())
                .limit(limit)
            )
            
            similarities = []
            for row in result:
                # Get the other recording ID
                other_id = row.recording_id_b if row.recording_id_a == recording_id else row.recording_id_a
                similarities.append((str(other_id), float(row.similarity_score)))
            
            return similarities
    
    async def batch_store_similarities(
        self,
        similarities: List[Tuple[str, str, float]],
        model_name: str = "mert-v1-95M"
    ) -> None:
        """Batch store multiple similarity scores."""
        async with self.get_session() as session:
            similarity_objects = []
            
            for id_a, id_b, score in similarities:
                # Ensure consistent ordering
                if id_a > id_b:
                    id_a, id_b = id_b, id_a
                
                similarity_objects.append(
                    SimilarityScore(
                        recording_id_a=id_a,
                        recording_id_b=id_b,
                        model_name=model_name,
                        similarity_score=score
                    )
                )
            
            # Use merge or ON CONFLICT logic here for production
            session.add_all(similarity_objects)
            
            try:
                await session.commit()
                logger.info(f"Stored {len(similarity_objects)} similarity scores")
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to batch store similarities: {e}")
                raise
    
    async def clear_similarities(
        self,
        model_name: Optional[str] = None
    ) -> int:
        """Clear similarity scores, optionally for a specific model."""
        async with self.get_session() as session:
            query = delete(SimilarityScore)
            
            if model_name:
                query = query.where(SimilarityScore.model_name == model_name)
            
            result = await session.execute(query)
            await session.commit()
            
            logger.info(f"Cleared {result.rowcount} similarity scores")
            return result.rowcount