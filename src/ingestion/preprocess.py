"""
Document preprocessing: clean extracted text and split into meaningful
chunks, preserving document/page metadata so retrieval results can be
cited back to a specific document and page.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import get_settings
from .loader import LoadedDocument


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    file_name: str
    page_number: int
    text: str
    source_type: str


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_document(document: LoadedDocument) -> list[Chunk]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings["retrieval"]["chunk_size"],
        chunk_overlap=settings["retrieval"]["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for page in document.pages:
        cleaned = _clean_text(page.text)
        if not cleaned:
            continue

        for i, piece in enumerate(splitter.split_text(cleaned)):
            chunk_id = f"{document.doc_id}_p{page.page_number}_c{i}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    file_name=document.file_name,
                    page_number=page.page_number,
                    text=piece,
                    source_type=page.source_type,
                )
            )

    return chunks
