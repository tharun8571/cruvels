"""
Vector store build/load. Uses Chroma locally for Stage 1 — no external
infra required. Each chunk is stored with metadata (doc_id, file_name,
page_number, source_type, visibility) so results stay traceable back to a
source and filterable by the authorization layer in src/context.
"""
from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

from src.config import get_path
from src.ingestion.preprocess import Chunk
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)


def _get_chroma_client(persist_dir: str | Path) -> chromadb.ClientAPI:
    persist_str = str(persist_dir)
    Path(persist_str).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_str,
        settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True,
        ),
    )


def build_vectorstore(chunks: list[Chunk], visibility: str = "shared") -> Chroma:
    """Embed chunks and persist them to the local Chroma store.
    `visibility` tags every chunk (shared/client/firm/private) so retrieval
    can later filter by what the current user/session is authorized to see.
    """
    persist_dir = get_path("vectorstore_dir")
    client = _get_chroma_client(persist_dir)
    embeddings = get_embeddings()

    texts = [c.text for c in chunks]
    metadatas = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "file_name": c.file_name,
            "page_number": c.page_number,
            "source_type": c.source_type,
            "visibility": visibility,
        }
        for c in chunks
    ]
    ids = [c.chunk_id for c in chunks]

    logger.info("Embedding and persisting %d chunks to %s", len(chunks), persist_dir)
    store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        client=client,
        collection_name="briefly_stage1",
    )
    return store


def load_vectorstore() -> Chroma:
    persist_dir = get_path("vectorstore_dir")
    client = _get_chroma_client(persist_dir)
    embeddings = get_embeddings()
    return Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name="briefly_stage1",
    )
