"""SQLAlchemy models for PostgreSQL database."""

from sqlalchemy import (
    Column, String, Integer, Numeric, DateTime, ForeignKey, 
    JSON, LargeBinary, CheckConstraint, UniqueConstraint, Text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class Recording(Base):
    """Recording model representing audio files."""
    __tablename__ = 'recordings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    duration_seconds = Column(Numeric(10, 2))
    recording_date = Column(DateTime(timezone=True))
    release_date = Column(DateTime(timezone=True))
    audio_s3_key = Column(String(500))  # For future S3 integration
    audio_local_path = Column(String(500))  # Current local file path
    audio_format = Column(String(20))
    sample_rate = Column(Integer)
    bit_depth = Column(Integer)
    dataset_source = Column(String(50))
    original_filename = Column(String(255))
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    embeddings = relationship("Embedding", back_populates="recording", cascade="all, delete-orphan")
    similarity_scores_a = relationship(
        "SimilarityScore", 
        foreign_keys="SimilarityScore.recording_id_a",
        back_populates="recording_a",
        cascade="all, delete-orphan"
    )
    similarity_scores_b = relationship(
        "SimilarityScore", 
        foreign_keys="SimilarityScore.recording_id_b", 
        back_populates="recording_b",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Recording(id={self.id}, title='{self.title}')>"


class Embedding(Base):
    """Embedding model for storing ML features."""
    __tablename__ = 'embeddings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recording_id = Column(UUID(as_uuid=True), ForeignKey('recordings.id', ondelete='CASCADE'))
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    feature_type = Column(String(50))  # aggregated, segments, raw
    dimension = Column(Integer, nullable=False)
    extraction_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    extraction_params = Column(JSON, default={})
    
    # Relationships
    recording = relationship("Recording", back_populates="embeddings")
    chunks = relationship("EmbeddingChunk", back_populates="embedding", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('recording_id', 'model_name', 'model_version', 'feature_type'),
    )
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, recording_id={self.recording_id}, model={self.model_name})>"


class EmbeddingChunk(Base):
    """Embedding chunks for large vectors."""
    __tablename__ = 'embedding_chunks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    embedding_id = Column(UUID(as_uuid=True), ForeignKey('embeddings.id', ondelete='CASCADE'))
    chunk_index = Column(Integer, nullable=False)
    chunk_data = Column(LargeBinary, nullable=False)  # Compressed numpy array
    chunk_size = Column(Integer, nullable=False)
    
    # Relationships
    embedding = relationship("Embedding", back_populates="chunks")
    
    __table_args__ = (
        UniqueConstraint('embedding_id', 'chunk_index'),
    )
    
    def __repr__(self):
        return f"<EmbeddingChunk(id={self.id}, embedding_id={self.embedding_id}, index={self.chunk_index})>"


class SimilarityScore(Base):
    """Pre-computed similarity scores between recordings."""
    __tablename__ = 'similarity_scores'
    
    recording_id_a = Column(UUID(as_uuid=True), ForeignKey('recordings.id', ondelete='CASCADE'))
    recording_id_b = Column(UUID(as_uuid=True), ForeignKey('recordings.id', ondelete='CASCADE'))
    model_name = Column(String(100))
    similarity_score = Column(Numeric(5, 4))  # 0.0000 to 1.0000
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    recording_a = relationship("Recording", foreign_keys=[recording_id_a], back_populates="similarity_scores_a")
    recording_b = relationship("Recording", foreign_keys=[recording_id_b], back_populates="similarity_scores_b")
    
    __table_args__ = (
        CheckConstraint('recording_id_a < recording_id_b', name='check_ordered_recording_ids'),
        UniqueConstraint('recording_id_a', 'recording_id_b', 'model_name'),
    )
    
    def __repr__(self):
        return f"<SimilarityScore(a={self.recording_id_a}, b={self.recording_id_b}, score={self.similarity_score})>"


# Future models for extended functionality
class Composer(Base):
    """Composer model for classical music metadata."""
    __tablename__ = 'composers'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    birth_year = Column(Integer)
    death_year = Column(Integer)
    nationality = Column(String(100))
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    works = relationship("Work", back_populates="composer")
    
    def __repr__(self):
        return f"<Composer(id={self.id}, name='{self.name}')>"


class Work(Base):
    """Musical work model."""
    __tablename__ = 'works'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    composer_id = Column(UUID(as_uuid=True), ForeignKey('composers.id'))
    title = Column(String(500), nullable=False)
    opus_number = Column(String(50))
    catalog_number = Column(String(50))  # BWV, K., etc.
    composition_year = Column(Integer)
    work_type = Column(String(100))  # symphony, concerto, sonata, etc.
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    composer = relationship("Composer", back_populates="works")
    recordings = relationship("Recording", back_populates="work")
    
    def __repr__(self):
        return f"<Work(id={self.id}, title='{self.title}')>"


class Tag(Base):
    """Tags for flexible categorization."""
    __tablename__ = 'tags'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tag_name = Column(String(100), unique=True, nullable=False)
    tag_category = Column(String(50))  # genre, mood, era, technique, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    recording_tags = relationship("RecordingTag", back_populates="tag")
    
    def __repr__(self):
        return f"<Tag(id={self.id}, name='{self.tag_name}')>"


class RecordingTag(Base):
    """Many-to-many relationship between recordings and tags."""
    __tablename__ = 'recording_tags'
    
    recording_id = Column(UUID(as_uuid=True), ForeignKey('recordings.id', ondelete='CASCADE'), primary_key=True)
    tag_id = Column(UUID(as_uuid=True), ForeignKey('tags.id'), primary_key=True)
    confidence = Column(Numeric(3, 2), default=1.0)  # 0.00 to 1.00
    
    # Relationships
    recording = relationship("Recording")
    tag = relationship("Tag", back_populates="recording_tags")
    
    def __repr__(self):
        return f"<RecordingTag(recording_id={self.recording_id}, tag_id={self.tag_id})>"


# Add work relationship to Recording (optional, for future use)
Recording.work_id = Column(UUID(as_uuid=True), ForeignKey('works.id'))
Recording.work = relationship("Work", back_populates="recordings")