#!/usr/bin/env python3
"""
Migrate existing MERT features from file system to Aurora PostgreSQL database.

This script reads the existing .npy feature files from data/processed/features/aggregated/
and stores them in the Aurora database using the chunked embedding storage system.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional
import traceback

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mess_ai.database import (
    RecordingRepository, 
    EmbeddingRepository, 
    DatabaseMigrator,
    aurora_client
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureMigrator:
    """Migrates existing features from file system to database."""
    
    def __init__(self):
        self.recording_repo = RecordingRepository(aurora_client)
        self.embedding_repo = EmbeddingRepository(aurora_client)
        
        # Paths to existing data
        self.audio_dir = Path("data/smd/wav-44")
        self.features_dir = Path("data/processed/features/aggregated")
        
        # Migration stats
        self.stats = {
            'recordings_created': 0,
            'embeddings_stored': 0,
            'errors': 0,
            'skipped': 0
        }
    
    async def get_audio_metadata(self, audio_file: Path) -> Dict:
        """Extract metadata from audio file."""
        try:
            import soundfile as sf
            info = sf.info(str(audio_file))
            return {
                'sample_rate': info.samplerate,
                'duration_seconds': info.duration,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype
            }
        except Exception as e:
            logger.warning(f"Could not read audio metadata for {audio_file}: {e}")
            return {
                'sample_rate': 44100,  # Default for SMD dataset
                'duration_seconds': None,
                'channels': 2,
                'format': 'WAV',
                'subtype': 'PCM_16'
            }
    
    async def migrate_recording(self, audio_file: Path) -> Optional[str]:
        """Create a recording entry in the database."""
        try:
            # Get audio metadata
            metadata = await self.get_audio_metadata(audio_file)
            
            # Create recording
            recording_id = await self.recording_repo.create_recording(
                title=audio_file.stem,
                dataset_source="smd",
                original_filename=audio_file.name,
                duration_seconds=metadata.get('duration_seconds'),
                metadata={
                    'audio_format': 'wav',
                    'sample_rate': metadata.get('sample_rate', 44100),
                    'channels': metadata.get('channels', 2),
                    'migrated_from': str(audio_file),
                    'migration_timestamp': str(asyncio.get_event_loop().time())
                }
            )
            
            self.stats['recordings_created'] += 1
            logger.info(f"Created recording {recording_id} for {audio_file.name}")
            return recording_id
            
        except Exception as e:
            logger.error(f"Failed to create recording for {audio_file}: {e}")
            self.stats['errors'] += 1
            return None
    
    async def migrate_embedding(self, recording_id: str, feature_file: Path) -> bool:
        """Store embedding from .npy file."""
        try:
            # Load feature vector
            embedding_vector = np.load(feature_file)
            logger.info(f"Loaded embedding shape: {embedding_vector.shape} from {feature_file}")
            
            # Store in database with chunking
            embedding_id = await self.embedding_repo.store_embedding(
                recording_id=recording_id,
                embedding_vector=embedding_vector,
                model_name="mert-v1-95M",
                model_version="1.0",
                feature_type="aggregated",
                extraction_params={
                    'source_file': str(feature_file),
                    'original_shape': list(embedding_vector.shape),
                    'migrated_from_file': True
                }
            )
            
            self.stats['embeddings_stored'] += 1
            logger.info(f"Stored embedding {embedding_id} for recording {recording_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store embedding from {feature_file}: {e}")
            logger.error(traceback.format_exc())
            self.stats['errors'] += 1
            return False
    
    async def verify_migration(self, recording_id: str, original_file: Path) -> bool:
        """Verify that the migration was successful."""
        try:
            # Test retrieving the embedding
            retrieved = await self.embedding_repo.get_embedding(recording_id)
            
            if retrieved is None:
                logger.error(f"Could not retrieve embedding for {recording_id}")
                return False
            
            # Compare with original
            original = np.load(original_file)
            
            if not np.allclose(original, retrieved, rtol=1e-6):
                logger.error(f"Embedding mismatch for {recording_id}")
                return False
            
            logger.info(f"Verified embedding for {recording_id} matches original")
            return True
            
        except Exception as e:
            logger.error(f"Verification failed for {recording_id}: {e}")
            return False
    
    async def find_audio_feature_pairs(self) -> List[tuple]:
        """Find matching audio and feature file pairs."""
        pairs = []
        
        if not self.audio_dir.exists():
            logger.error(f"Audio directory not found: {self.audio_dir}")
            return pairs
        
        if not self.features_dir.exists():
            logger.error(f"Features directory not found: {self.features_dir}")
            return pairs
        
        # Find all audio files
        audio_files = list(self.audio_dir.glob("*.wav"))
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Match with feature files
        for audio_file in audio_files:
            feature_file = self.features_dir / f"{audio_file.stem}.npy"
            
            if feature_file.exists():
                pairs.append((audio_file, feature_file))
                logger.debug(f"Matched pair: {audio_file.name} -> {feature_file.name}")
            else:
                logger.warning(f"No feature file found for {audio_file.name}")
                self.stats['skipped'] += 1
        
        logger.info(f"Found {len(pairs)} audio-feature pairs to migrate")
        return pairs
    
    async def migrate_all(self, verify: bool = True) -> Dict:
        """Migrate all audio and feature files."""
        logger.info("Starting feature migration to database...")
        
        # Find all pairs to migrate
        pairs = await self.find_audio_feature_pairs()
        
        if not pairs:
            logger.error("No audio-feature pairs found to migrate")
            return self.stats
        
        # Process each pair
        for i, (audio_file, feature_file) in enumerate(pairs, 1):
            logger.info(f"\nProcessing {i}/{len(pairs)}: {audio_file.name}")
            
            try:
                # Create recording
                recording_id = await self.migrate_recording(audio_file)
                
                if not recording_id:
                    continue
                
                # Store embedding
                success = await self.migrate_embedding(recording_id, feature_file)
                
                if not success:
                    continue
                
                # Verify if requested
                if verify:
                    verified = await self.verify_migration(recording_id, feature_file)
                    if not verified:
                        logger.warning(f"Verification failed for {audio_file.name}")
                        self.stats['errors'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process {audio_file.name}: {e}")
                self.stats['errors'] += 1
                continue
        
        # Print final stats
        logger.info("\n" + "="*50)
        logger.info("MIGRATION COMPLETE")
        logger.info("="*50)
        logger.info(f"Recordings created: {self.stats['recordings_created']}")
        logger.info(f"Embeddings stored: {self.stats['embeddings_stored']}")
        logger.info(f"Files skipped: {self.stats['skipped']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        return self.stats


async def main():
    """Main migration function."""
    try:
        # First, ensure database schema exists
        logger.info("Checking database schema...")
        migrator = DatabaseMigrator()
        
        if not await migrator.check_connection():
            logger.error("Cannot connect to database. Check your environment variables:")
            logger.error("- AURORA_CLUSTER_ARN")
            logger.error("- AURORA_SECRET_ARN") 
            logger.error("- AWS_REGION")
            return False
        
        # Run schema migration if needed
        schema_success = await migrator.migrate()
        if not schema_success:
            logger.error("Database schema migration failed")
            return False
        
        # Run feature migration
        feature_migrator = FeatureMigrator()
        stats = await feature_migrator.migrate_all(verify=True)
        
        # Check for success
        if stats['errors'] == 0:
            logger.info("🎉 Migration completed successfully!")
            return True
        else:
            logger.warning(f"⚠️  Migration completed with {stats['errors']} errors")
            return False
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Run migration
    success = asyncio.run(main())
    
    if success:
        print("\n✅ Database migration completed successfully!")
        print("Your MERT features are now stored in Aurora PostgreSQL.")
    else:
        print("\n❌ Migration failed. Check the logs above for details.")
        sys.exit(1)