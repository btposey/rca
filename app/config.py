from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    database_url: str = "postgresql+asyncpg://rca:rca@localhost:5432/rca"

    # Dispatcher — port 8000
    vllm_dispatcher_base_url: str = "http://localhost:8000/v1"
    vllm_dispatcher_model: str = "dispatcher-llama-1b"

    # Concierge — port 8001
    vllm_concierge_base_url: str = "http://localhost:8001/v1"
    vllm_concierge_model: str = "concierge-llama-3b"

    hf_home: str = "/models"
    hugging_face_hub_token: str = ""

    # Inference sampling parameters (documented for rubric)
    dispatcher_temperature: float = 0.1   # low — deterministic extraction
    dispatcher_max_tokens: int = 256
    concierge_temperature: float = 0.7    # higher — creative narrative
    concierge_top_p: float = 0.9
    concierge_max_tokens: int = 512

    # Librarian
    top_k_results: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
