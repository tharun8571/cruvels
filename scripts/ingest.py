"""
CLI: ingest every supported file in data/raw into the vector store.

Usage:
    python scripts/ingest.py --visibility shared
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings, get_path, setup_logging
from src.ingestion.loader import load_document
from src.ingestion.preprocess import chunk_document
from src.retrieval.vectorstore import build_vectorstore

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--visibility", default="shared", choices=["shared", "client", "firm", "private"])
    args = parser.parse_args()

    settings = get_settings()
    raw_dir = get_path("raw_data_dir")
    supported = settings["ingestion"]["supported_extensions"]

    files = [p for p in raw_dir.iterdir() if p.suffix.lower() in supported]
    if not files:
        logger.warning("No supported files found in %s", raw_dir)
        return

    all_chunks = []
    for file_path in files:
        document = load_document(file_path)
        chunks = chunk_document(document)
        logger.info("%s -> %d chunks", file_path.name, len(chunks))
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("No text extracted from any document; nothing to embed.")
        return

    build_vectorstore(all_chunks, visibility=args.visibility)
    logger.info("Ingestion complete: %d chunks from %d documents.", len(all_chunks), len(files))


if __name__ == "__main__":
    main()
