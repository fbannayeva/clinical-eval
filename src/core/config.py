from dotenv import load_dotenv
load_dotenv(override=True)

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    anthropic_api_key: str = Field(..., description="Anthropic Claude API key")
    llm_model: str = "claude-haiku-4-5-20251001"
    llm_max_tokens: int = 2048

    # Vector DB (Qdrant)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "clinical_trials"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # PostgreSQL / SQLite
    postgres_dsn: str = "sqlite+aiosqlite:///./cti.db"

    # External APIs
    pubmed_api_key: str | None = None
    clinicaltrials_base_url: str = "https://clinicaltrials.gov/api/v2"

    # Observability (Langfuse)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # App
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()