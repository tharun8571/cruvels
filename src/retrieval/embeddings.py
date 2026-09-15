"""
Embeddings factory. Kept separate from the vector store so the embedding
model can be swapped without touching retrieval logic.

Two HuggingFace modes supported:
  - provider: huggingface        → local model (sentence-transformers, for dev)
  - provider: huggingface-api   → HuggingFace Inference API (for Vercel, no sentence-transformers needed)
"""
from __future__ import annotations

import os

from src.config import get_settings


def get_embeddings():
    settings = get_settings()
    provider = settings["embeddings"]["provider"]
    model = settings["embeddings"]["model"]

    # ── Local HuggingFace (sentence-transformers) ──────────────────────────
    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
            return HuggingFaceEmbeddings(model_name=model)
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for provider='huggingface'. "
                "Run: pip install sentence-transformers  "
                "OR set EMBEDDING_PROVIDER=huggingface-api to use the free HF Inference API instead."
            ) from exc

    # ── HuggingFace Inference API (no local model, works on Vercel) ────────
    if provider == "huggingface-api":
        try:
            from langchain_huggingface import HuggingFaceEndpointEmbeddings  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "langchain-huggingface is required for provider='huggingface-api'. "
                "Run: pip install langchain-huggingface"
            ) from exc

        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise EnvironmentError(
                "HF_TOKEN env variable is required for provider='huggingface-api'. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        return HuggingFaceEndpointEmbeddings(
            model=model,
            huggingfacehub_api_token=hf_token,
        )

    # ── OpenAI-compatible (fallback) ───────────────────────────────────────
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
        return OpenAIEmbeddings(model=model)

    raise ValueError(
        f"Unsupported embeddings provider: '{provider}'. "
        "Choose from: huggingface, huggingface-api, openai"
    )
