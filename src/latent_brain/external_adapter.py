"""Public, neutral interface contract for a private agent's verifier.

Not a plugin for any specific controller. Nothing here can call tools, mutate
workspaces or manufacture trustworthy external execution evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .cookies import EVENT_COOKIES, DebugShadowAdapter

REWARD_TYPES = {**EVENT_COOKIES, 'hypothesis_refuted': 1}


@dataclass(frozen=True)
class ExternalEvidence:
    task_id: str
    obligation_id: str
    run_id: str
    workspace_revision: str
    evidence_ref: str
    kind: str
    # Metadata are only assertions until the host verifier checks them.
    result: str
    baseline_failure_ref: str
    origin: str


class TrustedRewardGate:
    """Only host-owned callbacks can validate independent external outcomes.

    There is no public HTTP reward endpoint. A caller that controls `verify`
    controls all rewards; provide only a real host-owned verifier callback.
    Agent supplied strings, receipts or claimed PASS do not count as proof.
    """

    def __init__(self, verify: Callable[[ExternalEvidence], bool]):
        if not callable(verify):
            raise ValueError('independent verifier is required')
        self._verify = verify
        self._awarded: set[tuple[str, str, str]] = set()
        self.total = 0

    def submit(self, event: ExternalEvidence) -> int:
        if not isinstance(event, ExternalEvidence):
            return 0
        attrs = (event.task_id, event.obligation_id, event.run_id,
                 event.workspace_revision, event.evidence_ref, event.kind,
                 event.result, event.baseline_failure_ref, event.origin)
        if any(not isinstance(x, str) or not x.strip() or len(x) > 256 for x in attrs):
            return 0
        if event.kind not in REWARD_TYPES or event.result != 'verified' or event.origin != 'preexisting':
            return 0
        key = (event.task_id, event.obligation_id, event.kind)
        if key in self._awarded:
            return 0
        if event.kind == 'case_resolved' and (event.task_id, event.obligation_id, 'hypothesis_confirmed') not in self._awarded:
            return 0
        # Both original baseline and latest workspace revision MUST be checked
        # by this independent host verifier; these fields alone are not proof.
        try:
            verified = self._verify(event) is True
        except Exception:
            return 0
        if not verified:
            return 0
        self._awarded.add(key)
        gained = REWARD_TYPES[event.kind]
        self.total += gained
        return gained
