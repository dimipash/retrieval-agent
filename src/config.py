from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str

    embedding_model: str = "all-MiniLM-L6-v2"

    chroma_persist_dir: str = "data/processed"
    chroma_collection: str = "retrieval_agent"

    chunk_size: int = Field(512, gt=0)
    chunk_overlap: int = Field(64, ge=0)

    top_k: int = Field(5, gt=0)


settings = Settings()
