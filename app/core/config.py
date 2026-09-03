from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "satquery-secret-key-default-dev"

    # Backend mode: 'mock' (default for testing without GPU/services) or 'http'
    MODEL_BACKEND: Literal["mock", "http"] = "mock"

    # Groq LLM Settings
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-70b-versatile"
    LLM_PROVIDER: Literal["groq", "mock"] = "mock"

    # Hosted Model Endpoints
    GEOCHAT_ENDPOINT: str = "http://localhost:8001/v1/predict"
    CHANGECHAT_ENDPOINT: str = "http://localhost:8002/v1/predict"
    PRITHVI_ENDPOINT: str = "http://localhost:8003/v1/predict"
    SAR_FUSION_ENDPOINT: str = "http://localhost:8004/v1/predict"
    BIGEARTHNET_ENDPOINT: str = "http://localhost:8005/bigearthnet_predict"

    # Timeout for HTTP model inference requests (in seconds)
    MODEL_HTTP_TIMEOUT: float = 60.0

    # Storage and DB
    DATABASE_URL: str = "sqlite+aiosqlite:///./satquery.db"
    STORAGE_PATH: str = "./storage"

    # Limits
    MAX_UPLOAD_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: set[str] = {
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
        ".npy",
    }


settings = Settings()
