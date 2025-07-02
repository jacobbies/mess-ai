#!/usr/bin/env python3
"""
Test script for Aurora PostgreSQL database integration.

This script tests the database connection, schema, and basic operations
to ensure everything is working correctly.
"""

import asyncio
import sys
import logging
from pathlib import Path
import numpy as np
import traceback

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mess_ai.database import (
    RecordingRepository,
    EmbeddingRepository, 
    SimilarityRepository,
    DatabaseMigrator,
    aurora_client
)
from src.mess_ai.database.config import check_database_setup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseTester:
    """Test suite for database functionality."""
    
    def __init__(self):
        self.recording_repo = RecordingRepository(aurora_client)
        self.embedding_repo = EmbeddingRepository(aurora_client)
        self.similarity_repo = SimilarityRepository(aurora_client)
        self.migrator = DatabaseMigrator(aurora_client)
        
        self.test_results = []
        
    def log_test_result(self, test_name: str, success: bool, message: str = ""):
        """Log and store test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        full_message = f"{status} {test_name}"
        if message:
            full_message += f": {message}"
        
        logger.info(full_message)
        self.test_results.append({
            'test': test_name,
            'success': success,
            'message': message
        })
        
        return success
    
    async def test_connection(self) -> bool:
        """Test basic database connection."""
        try:
            connected = await self.migrator.check_connection()
            return self.log_test_result(
                "Database Connection",
                connected,
                "Connected successfully" if connected else "Connection failed"
            )
        except Exception as e:
            return self.log_test_result(
                "Database Connection",
                False,
                f"Exception: {e}"
            )
    
    async def test_schema_exists(self) -> bool:
        """Test that required tables exist."""
        try:
            table_status = await self.migrator.check_tables_exist()
            all_exist = all(table_status.values())
            
            return self.log_test_result(
                "Schema Exists",
                all_exist,
                f"Tables: {table_status}"
            )
        except Exception as e:
            return self.log_test_result(
                "Schema Exists",
                False,
                f"Exception: {e}"
            )
    
    async def test_create_recording(self) -> tuple[bool, str]:
        """Test creating a recording."""
        try:
            recording_id = await self.recording_repo.create_recording(
                title="Test Recording",
                dataset_source="test",
                original_filename="test_file.wav",
                duration_seconds=120.5,
                metadata={"test": True, "sample_rate": 44100}
            )
            
            success = recording_id is not None
            return self.log_test_result(
                "Create Recording",
                success,
                f"Created recording {recording_id}" if success else "Failed to create recording"
            ), recording_id
            
        except Exception as e:
            return self.log_test_result(
                "Create Recording",
                False,
                f"Exception: {e}"
            ), None
    
    async def test_retrieve_recording(self, recording_id: str) -> bool:
        """Test retrieving a recording."""
        try:
            recording = await self.recording_repo.get_recording(recording_id)
            success = recording is not None and recording['id'] == recording_id
            
            return self.log_test_result(
                "Retrieve Recording",
                success,
                f"Retrieved {recording['title']}" if success else "Failed to retrieve recording"
            )
        except Exception as e:
            return self.log_test_result(
                "Retrieve Recording",
                False,
                f"Exception: {e}"
            )
    
    async def test_store_embedding(self, recording_id: str) -> tuple[bool, str]:
        """Test storing an embedding."""
        try:
            # Create a test embedding (13 x 768 like MERT aggregated features)
            test_embedding = np.random.randn(13, 768).astype(np.float32)
            
            embedding_id = await self.embedding_repo.store_embedding(
                recording_id=recording_id,
                embedding_vector=test_embedding,
                model_name="test-model",
                feature_type="test"
            )
            
            success = embedding_id is not None
            return self.log_test_result(
                "Store Embedding",
                success,
                f"Stored embedding {embedding_id}" if success else "Failed to store embedding"
            ), embedding_id
            
        except Exception as e:
            return self.log_test_result(
                "Store Embedding", 
                False,
                f"Exception: {e}"
            ), None
    
    async def test_retrieve_embedding(self, recording_id: str) -> bool:
        """Test retrieving an embedding."""
        try:
            embedding = await self.embedding_repo.get_embedding(
                recording_id=recording_id,
                model_name="test-model",
                feature_type="test"
            )
            
            success = embedding is not None and embedding.shape == (13, 768)
            
            return self.log_test_result(
                "Retrieve Embedding",
                success,
                f"Retrieved embedding shape {embedding.shape}" if success else "Failed to retrieve embedding"
            )
        except Exception as e:
            return self.log_test_result(
                "Retrieve Embedding",
                False,
                f"Exception: {e}"
            )
    
    async def test_similarity_storage(self, recording_id: str) -> bool:
        """Test similarity score storage and retrieval."""
        try:
            # Create a second test recording for similarity
            recording_id_2 = await self.recording_repo.create_recording(
                title="Test Recording 2",
                dataset_source="test", 
                original_filename="test_file_2.wav"
            )
            
            # Store similarity score
            await self.similarity_repo.store_similarity(
                recording_id_a=recording_id,
                recording_id_b=recording_id_2,
                similarity_score=0.85,
                model_name="test-model"
            )
            
            # Retrieve similar recordings
            similar = await self.similarity_repo.get_similar_recordings(
                recording_id=recording_id,
                model_name="test-model",
                limit=5
            )
            
            success = len(similar) > 0 and similar[0][0] == recording_id_2
            
            return self.log_test_result(
                "Similarity Storage",
                success,
                f"Found {len(similar)} similar recordings" if success else "Failed to store/retrieve similarities"
            )
            
        except Exception as e:
            return self.log_test_result(
                "Similarity Storage",
                False,
                f"Exception: {e}"
            )
    
    async def test_search_functionality(self) -> bool:
        """Test search functionality."""
        try:
            # Search for our test recordings
            results = await self.recording_repo.search_recordings("Test Recording")
            
            success = len(results) >= 1
            
            return self.log_test_result(
                "Search Functionality",
                success,
                f"Found {len(results)} search results" if success else "Search returned no results"
            )
        except Exception as e:
            return self.log_test_result(
                "Search Functionality",
                False,
                f"Exception: {e}"
            )
    
    async def test_cleanup(self) -> bool:
        """Clean up test data."""
        try:
            # Clean up test recordings
            await aurora_client.execute(
                "DELETE FROM recordings WHERE dataset_source = 'test'"
            )
            
            return self.log_test_result(
                "Cleanup",
                True,
                "Test data cleaned up successfully"
            )
        except Exception as e:
            return self.log_test_result(
                "Cleanup",
                False,
                f"Exception: {e}"
            )
    
    async def run_all_tests(self) -> bool:
        """Run the complete test suite."""
        logger.info("🧪 Starting database integration tests...")
        logger.info("=" * 50)
        
        # Test 1: Connection
        if not await self.test_connection():
            return False
        
        # Test 2: Schema
        if not await self.test_schema_exists():
            return False
        
        # Test 3: Create recording
        success, recording_id = await self.test_create_recording()
        if not success or not recording_id:
            return False
        
        # Test 4: Retrieve recording
        if not await self.test_retrieve_recording(recording_id):
            return False
        
        # Test 5: Store embedding
        success, embedding_id = await self.test_store_embedding(recording_id)
        if not success:
            return False
        
        # Test 6: Retrieve embedding
        if not await self.test_retrieve_embedding(recording_id):
            return False
        
        # Test 7: Similarity storage
        if not await self.test_similarity_storage(recording_id):
            return False
        
        # Test 8: Search functionality
        if not await self.test_search_functionality():
            return False
        
        # Test 9: Cleanup
        await self.test_cleanup()
        
        # Summary
        logger.info("=" * 50)
        passed = sum(1 for result in self.test_results if result['success'])
        total = len(self.test_results)
        
        if passed == total:
            logger.info(f"🎉 All {total} tests passed!")
            return True
        else:
            logger.error(f"❌ {total - passed} of {total} tests failed")
            return False


async def main():
    """Main test function."""
    try:
        # Check configuration first
        logger.info("Checking database configuration...")
        is_ready, message = check_database_setup()
        
        if not is_ready:
            logger.error("Database not properly configured:")
            logger.error(message)
            return False
        
        logger.info("✅ Database configuration looks good")
        
        # Run tests
        tester = DatabaseTester()
        success = await tester.run_all_tests()
        
        if success:
            logger.info("\n🎉 All database tests passed!")
            logger.info("Your Aurora PostgreSQL integration is working correctly.")
        else:
            logger.error("\n❌ Some tests failed. Check the logs above for details.")
        
        return success
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(main())
    
    if not success:
        sys.exit(1)