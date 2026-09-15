"""
Embeddings factory. Kept separate from the vector store so the embedding
model can be swapped (e.g. local model instead of OpenAI) without touching
retrieval logic.
"""
from __future__ import annotations

from src.config import get_settings


def get_embeddings():
    settings = get_settings()
    provider = settings["embeddings"]["provider"]
    model = settings["embeddings"]["model"]

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
            return HuggingFaceEmbeddings(model_name=model)
        except ImportError as exc:
            raise ImportError(
                "langchain-huggingface and sentence-transformers are required for "
                "the 'huggingface' embeddings provider. "
                "Install them or set EMBEDDING_PROVIDER=openai in your environment."
            ) from exc

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
        return OpenAIEmbeddings(model=model)

    raise ValueError(f"Unsupported embeddings provider: {provider}")
