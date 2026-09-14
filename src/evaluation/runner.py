"""
Evaluation runner: loads the evaluation dataset, runs each question through
the agent, and scores the results. This produces the test report required
in Stage 1 deliverables (accuracy, hallucination/fallback handling, retrieval
quality via citation presence).
"""
from __future__ import annotations

import json
import logging

from src.config import get_path
from src.context.permissions import AuthorizedScope
from src.agent.graph import run_agent
from .metrics import CaseResult, summarize_results

logger = logging.getLogger(__name__)


def run_evaluation(scope: AuthorizedScope | None = None) -> dict:
    dataset_path = get_path("eval_dataset")
    with open(dataset_path, "r") as f:
        cases = json.load(f)

    scope = scope or AuthorizedScope.firm_member(user_id="eval-user", case_id="eval-case")

    results: list[CaseResult] = []
    for case in cases:
        outcome = run_agent(case["question"], scope)
        results.append(
            CaseResult(
                question=case["question"],
                expected_answer_contains=case.get("expected_answer_contains", []),
                expected_fallback=case.get("expected_fallback", False),
                actual_answer=outcome["answer"],
                actual_sources=outcome["sources"],
                is_fallback=outcome["is_fallback"],
            )
        )
        logger.info("Evaluated: %s", case["question"])

    return summarize_results(results)


if __name__ == "__main__":
    import pprint

    pprint.pprint(run_evaluation())
