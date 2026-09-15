"""
Central config loader. Every layer imports `get_settings()` instead of
reading YAML directly, so config parsing happens once and stays in one place.
"""
from __future__ import annotations

import os
import logging
import logging.config
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

# On Vercel only /tmp is writable — remap data dirs automatically
_ON_VERCEL = os.getenv("VERCEL") == "1" or os.getenv("VERCEL_ENV") is not None
if _ON_VERCEL:
    os.environ.setdefault("VECTORSTORE_DIR", "/tmp/vectorstore")
    os.environ.setdefault("RAW_DATA_DIR", "/tmp/data/raw")
    os.environ.setdefault("PROCESSED_DATA_DIR", "/tmp/data/processed")


@lru_cache(maxsize=1)
def get_settings() -> dict:
    config_path = ROOT_DIR / "configs" / "settings.yaml"
    with open(config_path, "r") as f:
        settings = yaml.safe_load(f)

    # env overrides for anything secret / deployment-specific
    settings["llm"]["provider"] = os.getenv("LLM_PROVIDER", settings["llm"]["provider"])
    settings["llm"]["model"] = os.getenv("LLM_MODEL", settings["llm"]["model"])
    settings["embeddings"]["model"] = os.getenv(
        "EMBEDDING_MODEL", settings["embeddings"]["model"]
    )
    settings["embeddings"]["provider"] = os.getenv(
        "EMBEDDING_PROVIDER", settings["embeddings"]["provider"]
    )
    # Allow env overrides for paths (critical for Vercel /tmp remapping)
    if os.getenv("VECTORSTORE_DIR"):
        settings["paths"]["vectorstore_dir"] = os.getenv("VECTORSTORE_DIR")
    if os.getenv("RAW_DATA_DIR"):
        settings["paths"]["raw_data_dir"] = os.getenv("RAW_DATA_DIR")
    if os.getenv("PROCESSED_DATA_DIR"):
        settings["paths"]["processed_data_dir"] = os.getenv("PROCESSED_DATA_DIR")

    return settings


def setup_logging() -> None:
    log_config_path = ROOT_DIR / "configs" / "logging.yaml"
    with open(log_config_path, "r") as f:
        log_config = yaml.safe_load(f)
    logging.config.dictConfig(log_config)


def get_path(key: str) -> Path:
    """Resolve a path from settings['paths'] relative to project root,
    unless the value is already absolute (e.g. /tmp on Vercel)."""
    settings = get_settings()
    val = settings["paths"][key]
    p = Path(val)
    if p.is_absolute():
        return p.resolve()
    return (ROOT_DIR / val).resolve()
