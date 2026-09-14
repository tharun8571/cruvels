"""
LangGraph wiring for the Legal Knowledge Assistant.

Flow:
  agent -> [has tool calls? and under budget?]
              -> yes: tools (ToolNode) -> count_tool_call -> agent (loop)
              -> no:  finalize -> END

This replaces a version of the graph that called retrieval directly from a
node. Now the LLM is bound to real tools (document_search,
document_metadata_lookup) via a proper `ToolNode`, and decides for itself
whether/when to call them -- matching the assignment's "agent that can
choose between a limited number of tools" requirement (section 4.1) rather
than a fixed retrieval step.

The agent never takes any action beyond these two read-only tools -- no
external legal actions, no irreversible decisions (assignment section 4.2).
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.context.permissions import AuthorizedScope
from .state import AgentState
from .nodes import agent_node, should_continue, count_tool_call, finalize, get_tools_for_scope

logger = logging.getLogger(__name__)


def build_agent_graph(scope: AuthorizedScope):
    """Builds the graph bound to a specific AuthorizedScope. The tool node
    needs concrete, already-scoped tool instances (each tool closes over
    the scope so retrieval can never cross a visibility boundary), so the
    graph is built per-scope rather than once globally."""
    tools = get_tools_for_scope(scope)
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("count_tool_call", count_tool_call)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "count_tool_call")
    graph.add_edge("count_tool_call", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_agent(question: str, scope: AuthorizedScope) -> dict:
    """Entry point for running one question through the Legal Knowledge
    Assistant. Returns a dict with `answer`, `sources`, and `is_fallback`."""
    app = build_agent_graph(scope)
    initial_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "scope": scope,
        "tool_calls_made": 0,
    }
    final_state = app.invoke(initial_state)
    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "is_fallback": final_state.get("is_fallback", False),
    }
