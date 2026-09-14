"""
CLI: ask the Legal Knowledge Assistant a question against ingested docs.

Usage:
    python scripts/ask.py "What is the termination clause in the NDA?"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_logging
from src.context.permissions import AuthorizedScope
from src.agent.graph import run_agent


def main():
    setup_logging()
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "your question"')
        return

    question = sys.argv[1]
    scope = AuthorizedScope.firm_member(user_id="dev-user", case_id="dev-case")

    result = run_agent(question, scope)

    print("\nANSWER:\n" + result["answer"])
    if result["sources"]:
        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  - {s['file_name']} (p.{s['page_number']})")
    if result["is_fallback"]:
        print("\n[fallback: insufficient evidence]")


if __name__ == "__main__":
    main()
