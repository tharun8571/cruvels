"""
Shared state passed between LangGraph nodes. `messages` is the running
conversation (system prompt, human question, AI tool-call turns, tool
results) using LangGraph's `add_messages` reducer so each node only
returns the new messages it produced, not the whole history.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.context.permissions import AuthorizedScope


class SourceRef(TypedDict):
    file_name: str
    page_number: int
    doc_id: str


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    scope: AuthorizedScope

    tool_calls_made: int

    answer: str
    sources: list[SourceRef]
    is_fallback: bool
