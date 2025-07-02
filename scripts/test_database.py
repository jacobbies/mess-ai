#!/usr/bin/env python3
"""Test script for database integration."""
import asyncio
import sys
import os
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mess_ai.database import (
    aurora_client,
    RecordingRepository,
    EmbeddingRepository,
    SimilarityRepository,
    DatabaseMigrator
)

async def test_database_connection():
    """Test basic database connectivity."""
    print("🧪 Testing database connection...")
    
    try:
        result = await aurora_client.execute("SELECT NOW() as current_time, 'Hello Aurora!' as message")
        print(f"✅ Connected! Current time: {result[0]['current_time']}")
        print(f"✅ Message: {result[0]['message']}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

async def test_migration():
    """Test database migration."""
    print("\n🏗️ Testing database migration...")
    
    migrator = DatabaseMigrator()
    success = await migrator.migrate()
    
    if success:
        print("✅ Migration completed successfully")
        
        # Show table counts
        counts = await migrator.get_table_counts()
        for table, count in counts.items():
            print(f"  📊 {table}: {count} rows")
        
        return True
    else:
        print("❌ Migration failed")
        return False

async def test_recording_operations():
    """Test recording repository operations."""
    print("\n📀 Testing recording operations...")
    
    repo = RecordingRepository(aurora_client)
    
    try:
        # Create a test recording
        recording_id = await repo.create_recording(
            title="Test Classical Piece",
            dataset_source="test",
            original_filename="test_piece.wav",
            audio_s3_key="s3://test-bucket/test_piece.wav",
            duration_seconds=180.5,
            metadata={
                "composer": "Test Composer",
                "year": 2024,
                "genre": "classical"
            }
        )
        print(f"✅ Created recording: {recording_id}")
        
        # Retrieve the recording
        recording = await repo.get_recording(recording_id)
        if recording:
            print(f"✅ Retrieved recording: {recording['title']}")
            print(f"   Duration: {recording['duration_seconds']}s")
            print(f"   Metadata: {recording['metadata']}")
        else:
            print("❌ Failed to retrieve recording")
            return False
        
        # List recordings
        recordings = await repo.list_recordings(limit=5)
        print(f"✅ Listed {len(recordings)} recordings")
        
        # Search recordings
        search_results = await repo.search_recordings("Test", limit=5)
        print(f"✅ Search found {len(search_results)} results")
        
        return recording_id
        
    except Exception as e:
        print(f"❌ Recording operations failed: {e}")
        return None

async def test_embedding_operations(recording_id: str):
    """Test embedding repository operations."""
    print("\n🧠 Testing embedding operations...")
    
    repo = EmbeddingRepository(aurora_client)
    
    try:
        # Create a test embedding (simulating MERT output)
        test_embedding = np.random.rand(13, 768).astype(np.float32)  # MERT-like shape
        
        embedding_id = await repo.store_embedding(
            recording_id=recording_id,
            embedding_vector=test_embedding,
            model_name="mert-v1-95M",
            feature_type="aggregated",
            extraction_params={
                "segment_duration": 5.0,
                "overlap_ratio": 0.5
            }
        )
        print(f"✅ Stored embedding: {embedding_id}")
        
        # Retrieve the embedding
        retrieved_embedding = await repo.get_embedding(
            recording_id=recording_id,
            model_name="mert-v1-95M",
            feature_type="aggregated"
        )
        
        if retrieved_embedding is not None:
            print(f"✅ Retrieved embedding shape: {retrieved_embedding.shape}")
            
            # Verify the data is correct
            if np.allclose(test_embedding, retrieved_embedding, rtol=1e-5):
                print("✅ Embedding data matches original")
            else:
                print("❌ Embedding data doesn't match original")
                return False
        else:
            print("❌ Failed to retrieve embedding")
            return False
        
        # List embeddings for the recording
        embeddings_list = await repo.list_embeddings(recording_id)
        print(f"✅ Found {len(embeddings_list)} embeddings for recording")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding operations failed: {e}")
        return False

async def test_similarity_operations(recording_id: str):
    """Test similarity repository operations."""
    print("\n🔍 Testing similarity operations...")
    
    repo = SimilarityRepository(aurora_client)
    recording_repo = RecordingRepository(aurora_client)
    
    try:
        # Create another test recording for similarity
        recording_id_2 = await recording_repo.create_recording(
            title="Another Test Piece",
            dataset_source="test",
            original_filename="test_piece_2.wav",
            metadata={"composer": "Another Composer"}
        )
        print(f"✅ Created second recording: {recording_id_2}")
        
        # Store a similarity score
        await repo.store_similarity(
            recording_id_a=recording_id,
            recording_id_b=recording_id_2,
            similarity_score=0.85,
            model_name="mert-v1-95M"
        )
        print("✅ Stored similarity score")
        
        # Retrieve similar recordings
        similar = await repo.get_similar_recordings(
            recording_id=recording_id,
            model_name="mert-v1-95M",
            limit=5
        )
        
        if similar:
            print(f"✅ Found {len(similar)} similar recordings")
            for similar_id, score in similar:
                print(f"   {similar_id}: {score:.3f}")
        else:
            print("❌ No similar recordings found")
        
        return True
        
    except Exception as e:
        print(f"❌ Similarity operations failed: {e}")
        return False

async def main():
    """Run all database tests."""
    print("🚀 Starting MESS-AI Database Tests\n")
    
    # Test connection
    if not await test_database_connection():
        print("\n❌ Database connection failed - check your AWS configuration")
        sys.exit(1)
    
    # Test migration
    if not await test_migration():
        print("\n❌ Database migration failed")
        sys.exit(1)
    
    # Test recording operations
    recording_id = await test_recording_operations()
    if not recording_id:
        print("\n❌ Recording operations failed")
        sys.exit(1)
    
    # Test embedding operations
    if not await test_embedding_operations(recording_id):
        print("\n❌ Embedding operations failed")
        sys.exit(1)
    
    # Test similarity operations
    if not await test_similarity_operations(recording_id):
        print("\n❌ Similarity operations failed")
        sys.exit(1)
    
    print("\n🎉 All database tests passed!")
    print("\n📋 Summary:")
    print("   ✅ Database connection")
    print("   ✅ Schema migration")
    print("   ✅ Recording CRUD operations")
    print("   ✅ Embedding storage/retrieval")
    print("   ✅ Similarity scoring")
    print("\n🚀 Database integration is ready!")

if __name__ == "__main__":
    asyncio.run(main())