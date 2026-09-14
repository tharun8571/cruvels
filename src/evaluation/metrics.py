"""
Evaluation metrics for Stage 1 acceptance criteria: does the system answer
correctly, cite sources, and fall back honestly when evidence is missing
-- rather than a single blended "accuracy" number that hides which of
those actually failed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaseResult:
    question: str
    expected_answer_contains: list[str]
    expected_fallback: bool
    actual_answer: str
    actual_sources: list[dict]
    is_fallback: bool

    @property
    def answer_match(self) -> bool:
        if self.expected_fallback:
            return self.is_fallback
        if not self.expected_answer_contains:
            return True
        lowered = self.actual_answer.lower()
        return any(kw.lower() in lowered for kw in self.expected_answer_contains)

    @property
    def has_citation(self) -> bool:
        return len(self.actual_sources) > 0 or self.is_fallback

    @property
    def fallback_correct(self) -> bool:
        return self.is_fallback == self.expected_fallback


def summarize_results(results: list[CaseResult]) -> dict:
    total = len(results)
    if total == 0:
        return {"total": 0}

    return {
        "total": total,
        "answer_match_rate": sum(r.answer_match for r in results) / total,
        "citation_rate": sum(r.has_citation for r in results) / total,
        "fallback_correct_rate": sum(r.fallback_correct for r in results) / total,
        "failures": [r.question for r in results if not r.answer_match or not r.fallback_correct],
    }
