from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field
from typing import List

class Settings(BaseSettings):

    APP_NAME: str = "Classical RecSys API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    #AWS Configuration
    AWS_REGION: str = Field(default="us-west-1", description="AWS region where s3 bucket is")
    S3_BUCKET_NAME: str = Field(default='mess-ai-1-bucket', 
                                description="S3 bucket containing embeddings, audio files, react files")
    
    #S3 Paths
    AUDIO_PREFIX: str = "audio"
    EMBEDDINGS_PREFIX: str = Field(default="embeddings/", description = "S3 prefix for embeddings files")
    AUDIO_PREFIX_MAESTRO: str = "maestro"
    AUDIO_PREFIX_SMD: str = "smd-wav44"

    #API Configuration
    DEFAULT_RECOMMENDATION_COUNT: int = 5
    MAX_RECOMMENDATION_COUNT: int = 10

    #Vercel + Frontend
    


settings = Settings()

