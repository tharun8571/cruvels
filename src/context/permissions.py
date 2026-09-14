"""
Authorization boundary. Enforces the case-record visibility model from the
product overview: shared / client / firm / private layers must never mix
across an unauthorized session. This is the single place that decides
whether a chunk is visible to the current user/session -- retrieval must
always go through it, never filter by visibility inline elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field

VISIBILITY_LEVELS = ("shared", "client", "firm", "private")


@dataclass
class AuthorizedScope:
    """Represents what a single request/session is allowed to see.

    user_id: identifies the requester, used for private-visibility checks.
    case_id: restricts retrieval to a single case record.
    allowed_visibilities: which visibility layers this session may read.
    allowed_doc_ids: optional explicit allow-list; if empty, any doc within
        the allowed visibilities and case is permitted.
    """

    user_id: str
    case_id: str
    allowed_visibilities: tuple[str, ...] = ("shared",)
    allowed_doc_ids: set[str] = field(default_factory=set)

    def can_access(self, visibility: str, doc_id: str | None) -> bool:
        if visibility not in VISIBILITY_LEVELS:
            return False
        if visibility not in self.allowed_visibilities:
            return False
        if self.allowed_doc_ids and doc_id not in self.allowed_doc_ids:
            return False
        return True

    @classmethod
    def shared_only(cls, user_id: str, case_id: str) -> "AuthorizedScope":
        return cls(user_id=user_id, case_id=case_id, allowed_visibilities=("shared",))

    @classmethod
    def firm_member(cls, user_id: str, case_id: str) -> "AuthorizedScope":
        return cls(user_id=user_id, case_id=case_id, allowed_visibilities=("shared", "firm", "client"))
