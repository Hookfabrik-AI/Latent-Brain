# Latent Brain

**A small, standalone, CPU-first neural world-model experiment.**

> Status: **research prototype v0.2**. Not an AGI, a general reasoning model, or a verified improvement to any LLM. This repository contains **no proprietary agent/controller code**.

The goal is to test a falsifiable claim: **Can a small network learn transition dynamics from verified experiences on training environments, generalize to unseen environments, and support model-based lookahead that improves actual task completion over a simple reactive heuristic?**

## What works in v0.2

- Deterministic, procedurally generated maze environments with real move/reward rules and held-out layouts.
- A small **recurrent GRU-based latent feature refiner**, trained to predict if a requested move will succeed.
- Model-based planning: **A\*** simulates predicted transitions, never calls the ground-truth transition engine during planning, and produces a candidate plan.
- Independent verification: the proposed plan is replayed in the **real** environment, and failed plans are counted as failures.
- Honest baselines: majority-class prediction, a one-step greedy agent with map access, and an oracle planning ceiling.
- Parameter count, training losses, held-out transition accuracy and actual navigation success are recorded.
- Local-only, versioned JSON HTTP API for inference and candidate planning (no extra web-framework dependency).

**Important limitation:** The hidden state is refined recurrently *within each prediction*, not maintained persistently across world steps. A\* supplies the explicit search algorithm; the neural model supplies transition predictions. This is **not yet autonomous learned reasoning**, variable-depth deliberation, online learning, or LLM integration. Those are future experiments, not existing features.

## Quick start

Python 3.10+; CPU only is sufficient. PyTorch wheels can be large.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m latent_brain train
python -m latent_brain evaluate
python -m latent_brain demo
```

Outputs: `checkpoints/brain.pt` and `reports/evaluation.json` (ignored by git by default). Evaluation automatically uses **held-out seed/layouts** stored in the checkpoint metadata, not an overlapping training split. No API keys are required. A small demonstration checkpoint is also included at `examples/pretrained_v0.1.pt` (see the published evaluation protocol; do not assume its metrics transfer to other tasks).

To run a quick demo using the included weights without first training, run `python -m latent_brain demo --checkpoint examples/pretrained_v0.1.pt`.

To experiment with hidden size and recurrent computation:

```bash
python -m latent_brain train --hidden-size 128 --thinking-steps 4 --epochs 12
python -m latent_brain evaluate
```

### Local HTTP API (v0.2)

The API loads an existing checkpoint **once** and serves CPU inference. It does not
train, contact an external LLM, or execute the actions it proposes.

```bash
python -m latent_brain serve --checkpoint examples/pretrained_v0.1.pt --port 8767
curl http://127.0.0.1:8767/v1/health
curl -s http://127.0.0.1:8767/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"rows":["...",".#.","..."],"queries":[{"state":[0,0],"action":"right"}]}'
curl -s http://127.0.0.1:8767/v1/plan \
  -H 'Content-Type: application/json' \
  -d '{"rows":["...",".#.","..."]}'
```

Only `127.0.0.1`/`localhost` is supported: no remote exposure or authentication
is implemented. The API's `verification: not_executed` output makes clear that
plans are hypotheses, **not proof of execution**. See [API documentation](docs/API.md).

### Results / proof standard

After executing the commands, use the **actual generated `reports/evaluation.json`**. We do not report benchmark wins unless a reproducible run demonstrates them. A successful test suite alone says nothing about learned planning quality. Rerun over multiple seeds before drawing conclusions; the simple greedy baseline is not a strong model-based planning comparison.

## Diagram

```text
known local state + action
          |
          v
  encoder + GRU refinement   <-- trained from verified transitions
          |
          v
  predicted move succeeds?
          |
          v
 A* hypothetical rollouts    <-- no calls to true transition rules
          |
          v
  proposed action sequence
          |
          v
 independent real replay     <-- source of truth
```

## Boundaries & privacy

Its public interface (`LatentBrain`, `brain_predictor`, `plan`) can be wrapped by private consumers outside this repository. If contributing traces, only submit data you have a right to distribute and that have been reviewed for personal/sensitive content and trade secrets.

## Future milestones (NOT implemented)

1. Predict multi-step latent dynamics; measure rollout error accumulation.
2. Add adaptive thinking-budget learning rather than fixed GRU iterations.
3. Learn from isolated, online experiments using real rewards rather than only supervised transitions.
4. Compare against **the same planner using a non-neural transition model** and stronger alternatives, across multiple unseen tasks and seeds.
5. Public neutral adapter for external coding LLMs; test same coder with vs without the model at equal budgets.

See [research protocol](docs/EXPERIMENTS.md), [reproducible results](docs/RESULTS.md), [contributing](CONTRIBUTING.md), and [security](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
