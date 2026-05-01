"""Central config — loaded once, imported everywhere."""
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API keys
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")       # free @ aistudio.google.com
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")   # free @ app.pinecone.io
    pinecone_env: str = Field("us-east-1", env="PINECONE_ENV")

    # Pinecone
    pinecone_index_name: str = Field("multimodal-rag", env="PINECONE_INDEX_NAME")
    pinecone_dimension: int = Field(768, env="PINECONE_DIMENSION")  # bge-base = 768 dims

    # Models — all free
    embed_model: str = Field("BAAI/bge-base-en-v1.5", env="EMBED_MODEL")                  # local
    vision_model: str = Field("gemini-2.5-flash", env="VISION_MODEL")                     # free API
    chat_model: str = Field("gemini-2.5-flash", env="CHAT_MODEL")                         # free API
    rerank_model: str = Field("cross-encoder/ms-marco-MiniLM-L6-v2", env="RERANK_MODEL")  # local

    # Retrieval
    top_k_retrieval: int = Field(20, env="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(5, env="TOP_K_RERANK")
    hybrid_alpha: float = Field(0.5, env="HYBRID_ALPHA")
    chunk_size: int = Field(800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(150, env="CHUNK_OVERLAP")

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()