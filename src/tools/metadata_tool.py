"""
Metadata lookup tool: lets the agent check which documents exist in the
authorized collection without pulling full content -- useful when the user
asks something like "what documents do we have" or "is there a contract
in this case" before doing a full semantic search.
"""
from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.context.permissions import AuthorizedScope
from src.retrieval.vectorstore import load_vectorstore


class MetadataLookupInput(BaseModel):
    file_name_contains: str = Field(
        default="",
        description="Optional substring to filter document names by. Leave empty to list all authorized documents.",
    )


def make_metadata_lookup_tool(scope: AuthorizedScope) -> StructuredTool:
    def _lookup(file_name_contains: str = "") -> str:
        store = load_vectorstore()
        raw = store.get(include=["metadatas"])
        metadatas = raw.get("metadatas", [])

        seen = {}
        for meta in metadatas:
            if not scope.can_access(meta.get("visibility", "shared"), meta.get("doc_id")):
                continue
            if file_name_contains and file_name_contains.lower() not in meta["file_name"].lower():
                continue
            seen[meta["doc_id"]] = meta["file_name"]

        docs = [{"doc_id": k, "file_name": v} for k, v in seen.items()]
        return json.dumps({"documents": docs})

    return StructuredTool.from_function(
        func=_lookup,
        name="document_metadata_lookup",
        description=(
            "List authorized documents in the current case, optionally filtered by file name substring. "
            "Use this to check what documents are available before searching their content."
        ),
        args_schema=MetadataLookupInput,
    )
