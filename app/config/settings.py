from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    # Embeddings (local)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    default_top_k: int = 5
    max_top_k: int = 20

    # Grounding
    max_retries: int = 2

    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # Data paths
    data_uploads_dir: Path = Path("data/uploads")
    data_index_dir: Path = Path("data/index")


settings = Settings()
