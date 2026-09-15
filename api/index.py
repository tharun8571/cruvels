# Vercel entrypoint – re-exports the FastAPI app from src/api/main.py
from src.api.main import app  # noqa: F401  – Vercel picks up `app` from here
