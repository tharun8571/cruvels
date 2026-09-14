"""
Document summarization. Kept separate from the Q&A agent graph since it's
a distinct Stage 1 feature (summarize a supplied document, not answer a
question against retrieved chunks) with its own prompt and no retrieval
step needed.
"""
from __future__ import annotations

from src.config import get_path
from src.llm.model import get_llm
from src.ingestion.loader import LoadedDocument

_PROMPTS_DIR = get_path("prompts_dir")


def summarize_document(document: LoadedDocument, max_chars: int = 12000) -> str:
    llm = get_llm()
    prompt_template = (_PROMPTS_DIR / "summarization_prompt.txt").read_text()

    text = document.full_text[:max_chars]
    prompt = prompt_template.format(document_text=text)

    response = llm.invoke(prompt)
    return response.content
