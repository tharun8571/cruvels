"""
Document search tool: the controlled function the agent calls to retrieve
evidence. This is a thin wrapper around src.retrieval.retriever so the
agent never talks to the vector store directly -- it only sees a tool
interface with a fixed input/output contract.
"""
from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.context.permissions import AuthorizedScope
from src.retrieval.retriever import retrieve_chunks


class DocumentSearchInput(BaseModel):
    query: str = Field(description="The search query to find relevant document sections")


def make_document_search_tool(scope: AuthorizedScope) -> StructuredTool:
    """Builds a document-search tool bound to a specific authorized scope,
    so the agent can never retrieve content outside what the current
    user/session is permitted to see."""

    def _search(query: str) -> str:
        results = retrieve_chunks(query, scope)
        if not results:
            return json.dumps({"results": [], "note": "No relevant authorized content found."})

        payload = [
            {
                "text": r.text,
                "doc_id": r.doc_id,
                "file_name": r.file_name,
                "page_number": r.page_number,
                "score": round(r.score, 3),
            }
            for r in results
        ]
        return json.dumps({"results": payload})

    return StructuredTool.from_function(
        func=_search,
        name="document_search",
        description=(
            "Search authorized case documents for content relevant to a query. "
            "Returns matching text chunks with source document name and page number. "
            "Use this before answering any factual question."
        ),
        args_schema=DocumentSearchInput,
    )
