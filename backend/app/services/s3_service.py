import boto3
import numpy as np
import logging
import io
from typing import List
from app.core.config import settings
from app.services.s3_keys import S3Keys

logger = logging.getLogger(__name__)

class S3Service:
    def __init__(self):
        self.bucket_name = settings.S3_BUCKET_NAME
        self.region = settings.AWS_REGION
        self.s3 = boto3.client('s3', region_name=self.region)
        logger.info(f"S3 Service initialized: {self.bucket_name}, {self.region}")

    def get_audio_preurl(self, audio_S3key: str, expiration: int = 600) -> str:
        """
            audio_path: S3 key
            expiration: URL valid for X seconds (default 5min)
        """
        try:    
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': audio_S3key
                },
                ExpiresIn=expiration
            )
            logger.debug(f"Generate presigned URL for {audio_S3key}")
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned url for {audio_S3key}: {e}")
            raise
    
    def list_embeddings(self, dataset: str = 'smd') -> List[str]:
        """
        List all embedding files for a dataset
        """
        prefix = S3Keys.embeddings_prefix_smd()

        embedding_files = []
        paginator = self.s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

        for page in page_iterator:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.npy'):
                    embedding_files.append(key)

        logger.info(f"Found {len(embedding_files)} embeddings for {dataset}")
        return embedding_files
    
    def download_embedding(self, embedding_S3key: str) -> np.ndarray:
        """
        Download a single embedding file (.npy) from S3, converts to np.ndarray in RAM
        """
        try:
            response=self.s3.get_object(
                Bucket=self.bucket_name,
                Key=embedding_S3key
            )
            bytes_data = response['Body'].read()
            embedding = np.load(io.BytesIO(bytes_data))

            return embedding
        except Exception as e:
            logger.error(f"Failed to download embedding {embedding_S3key}")
            raise

    def check_connection(self) -> bool:
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Connected to bucket: {self.bucket_name}")
            return True
        except Exception as e:
            logger.error(f"S3 connection failed: {e}")
            return False

    


if __name__ == '__main__':
    s3service = S3Service()

    track_id = "Bach_BWV849-01_001_20090916-SMD"

    # Test embedding download
    embedding_key = S3Keys.embedding_smd(track_id)
    embedding = s3service.download_embedding(embedding_key)
    print(f"Embedding shape: {embedding.shape}")

    # Test presigned URL generation
    audio_key = S3Keys.audio_smd(track_id)
    url = s3service.get_audio_preurl(audio_key)
    print(f"Audio URL: {url[:100]}...")

    # Test listing embeddings
    embeddings = s3service.list_embeddings('smd')
    print(f"Total embeddings: {len(embeddings)}")
        




    