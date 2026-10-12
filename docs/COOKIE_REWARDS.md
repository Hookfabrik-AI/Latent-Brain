# Verified Cookie Reward Learning · v0.4 (toy research)

**Status: toy research experiment, not a deployed coding agent.** This module has
no private runtime, no write privileges, no network dependency, and cannot edit
external source code or another project. A reward is not permission to execute.

## Architecture

A separate `RewardFlyBrain` uses the published FlyBrain's **GRU memory, four-way
policy, learned latent imagination, and 1/2/4 simulation-depth head**, adding a
state-value head. Its 18 observation fields represent synthetic debugging state,
**not** the maze observations used in v0.3, so the checkpoints are deliberately
incompatible. Actions are `inspect`, `test_a`, `test_b`, and `report`.

The `ToyDebugLab` is an independent environment with a hidden binary defect
hypothesis. It observes genuine *toy-environment* outcomes and issues HMAC
receipts to a private `CookieLedger`. Awards are one-time: inspect +1,
confirmed hypothesis +3, verified resolution +5. Invalid, duplicate or
self-reported receipts earn **zero**. Uninspected guesswork is refused by the
toy controller; incomplete completion claims are also rejected. Actions incur a small cost and incomplete
reports are rejected (not marked complete). Models cannot mint receipts.

The policy is updated with sampled, discounted **verified environment rewards**,
a learned value baseline, entropy regularization and latent prediction
consistency. No oracle/teacher action labels are used to train v0.4. The
simulation-budget head is trained through the same returns with a small
computation penalty. This does **not** establish optimal allocation of thought.

```text
Toy observation -> GRU memory -> budget head -> imagined action paths
                                    |                  |
                                    v                  v
                               simulated depth    policy decision
                                                      |
                         verifier-owned toy execution (not model)
                                                      |
                                    verified receipt + observed next state
                                                      |
                            single-use ledger -> reward -> policy update
```

## Reproduce

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python -m latent_brain train-cookie --seed 42 --episodes 250 --hidden-size 32 \
  --reward-checkpoint checkpoints/reward.pt
python -m latent_brain evaluate-cookie \
  --reward-checkpoint checkpoints/reward.pt --cases 80
python -m latent_brain demo-cookie \
  --reward-checkpoint checkpoints/reward.pt
```

Every evaluation replays decisions through an independent toy environment.
Do **not** relabel `toy_execution_verified` as code, sandbox or external source code verification.
Training and held-out evaluation use disjoint deterministic seed namespaces.

## Adapter and safety boundary

`DebugShadowAdapter` accepts an explicit 18-field numeric observation in [0,1].
It returns an **advisory-only**, **not-executed** proposal plus depth. The host
must own the observation mapping, real execution, revision binding, verifier,
reward ledger, task/episode reset and external training-data provenance.
It cannot ingest self-reported rewards or assert a PASS. Its per-task GRU state
must be reset between cases. It does **not** import private project components.

**No direct integration is claimed.** To connect a production agent, first obtain its
source and a real controller/verifier contract. Map inputs only to the neutral
public interface. External training data must be rights-cleared and sanitized
before being published. Do not ship private traces or agent code here.

## Acceptance criteria for a future coding-lab integration

- Host verifies every outcome against real run ID, project revision, obligation
  and immutable evidence; failure of any check means zero reward.
- No reward for duplicate IDs, replay, fabricated PASS, or introduced bugs.
- Agent never owns signing keys, independent verifier, cookies or authority.
- Read-only shadow comparison first, with held-out projects/tasks and equal
  budgets against **no reward learner**, fixed depth, and the existing agent.
- Reward-hacking probes, failure cases, and full execution provenance published
  separately (with sensitive/private data removed).
- Human approval required before an agent uses recommendations to modify code.

## Honest limitations

The toy lab is deliberately tiny; it teaches a short evidence workflow, not
Python debugging or reasoning across an actual source tree. On some seeds the
budget head chooses a single depth most of the time. Any measured policy lift
must not be described as proof of useful adaptive compute or external source code repair skill.

## Honest comparison

The executable `evaluate_rule_baseline` follows a simple documented rule:
inspect, test the revealed hypothesis, report. It solves this trivial toy
workflow without training. It is the meaningful competence ceiling here.
The neural policy learning it too is a **plumbing demonstration**, not a
claim of superior debugging or reasoning. Depth choices are measured, but
no reliable lift over forced depth 1 has been established.
