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


def is_serverless() -> bool:
    """Detects whether code is executing in a serverless environment (Vercel, AWS Lambda)
    where the root filesystem (/var/task) is strictly read-only."""
    return (
        os.getenv("VERCEL") is not None
        or os.getenv("VERCEL_ENV") is not None
        or os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None
        or os.getenv("LAMBDA_TASK_ROOT") is not None
        or str(ROOT_DIR).startswith("/var/task")
        or str(ROOT_DIR).startswith("/var/runtime")
        or not os.access(ROOT_DIR, os.W_OK)
    )


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

    # In serverless environments, redirect all writable paths to /tmp
    if is_serverless():
        settings["paths"]["raw_data_dir"] = "/tmp/data/raw"
        settings["paths"]["processed_data_dir"] = "/tmp/data/processed"
        settings["paths"]["vectorstore_dir"] = "/tmp/vectorstore"

    # Allow explicit env overrides for paths
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
    """Resolve a path from settings['paths'].
    Writable paths in serverless environments are guaranteed to be in /tmp.
    """
    settings = get_settings()
    val = settings["paths"][key]
    p = Path(val)
    if p.is_absolute():
        return p.resolve()
    return (ROOT_DIR / val).resolve()
