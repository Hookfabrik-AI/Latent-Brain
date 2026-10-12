from dataclasses import replace

from latent_brain.external_adapter import ExternalEvidence, TrustedRewardGate


def make_event(kind='inspected', **updates):
    obj = ExternalEvidence(task_id='task1', obligation_id='r1', run_id='run1',
                           workspace_revision='abc', evidence_ref='sha256:deadbeef',
                           kind=kind, result='verified', baseline_failure_ref='run0',
                           origin='preexisting')
    return replace(obj, **updates)


def test_no_external_rewards_without_real_verifier():
    gate = TrustedRewardGate(lambda event: False)
    assert gate.submit(make_event()) == 0
    assert gate.submit(make_event(result='self_report')) == 0
    assert gate.submit(make_event(origin='introduced_by_agent')) == 0
    assert gate.total == 0


def test_duplicate_task_reward_blocked_across_runs():
    seen = []
    gate = TrustedRewardGate(lambda e: seen.append(e) or True)
    assert gate.submit(make_event()) == 1
    assert gate.submit(make_event(run_id='run2',workspace_revision='def')) == 0
    assert gate.total == 1
    assert len(seen) == 1


def test_verified_outcomes_require_order_and_independent_proof():
    gate = TrustedRewardGate(lambda event: event.evidence_ref in {'e1', 'e2'})
    assert gate.submit(make_event(kind='case_resolved',evidence_ref='e2')) == 0
    assert gate.submit(make_event(kind='hypothesis_confirmed',evidence_ref='invalid')) == 0
    assert gate.submit(make_event(kind='hypothesis_confirmed',evidence_ref='e1')) == 3
    assert gate.submit(make_event(kind='case_resolved',evidence_ref='e2')) == 5
    assert gate.total == 8


def test_invalid_fields_or_host_verifier_exception_fail_closed():
    gate = TrustedRewardGate(lambda event: 1 / 0)
    assert gate.submit(make_event()) == 0
    assert gate.submit(make_event(workspace_revision='')) == 0
    assert gate.submit(make_event(evidence_ref='x'*257)) == 0
    assert gate.total == 0
