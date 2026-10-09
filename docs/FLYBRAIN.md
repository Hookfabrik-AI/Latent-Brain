# Experimental FlyBrain v0.3

This project contains an **independent, tiny, CPU-friendly** recurrent policy model.
It is not an LLM and has no execution permission. The 0.1 transition model and
0.2 API remain compatible; the FlyBrain checkpoint has its own format.

## Architecture

- `FlyBrain.encoder` + `GRUCell`: hidden state across observations in one episode.
- `policy`: learned scores for four candidate actions.
- `budget`: three learned logits choosing among 1, 2, or 4 simulated steps.
- `imagination`: learned next-hidden-state predictor from `(hidden, action)`.
- Separate pretrained transition predictor: estimates whether a maze move is possible.
- Bounded beam search: considers **hypothetical** transitions and candidate actions.
- Real `Maze.move()` is called **only during training teacher-label generation or
  independent evaluation**, not inside the deployed decision engine.

## Learning and its limitations

`train-fly` uses generated training mazes, verified shortest-path demonstrations,
and actual transitions for supervised data. Training minimizes action imitation,
budget classification from **heuristic difficulty proxies**, and a latent-step
consistency loss. This is not reinforcement learning, not free-form logical
reasoning and not a claim that budgets are optimal or saved compute.

No result on a held-out maze is available to the training loop. New evaluation
maps use a different seed and are checked for overlapping layouts.

## Held-out comparison

Three independent training/evaluation seeds; 30 held-out 7×7 mazes each.
Counts are real environment goal completions, not predicted successes.

| Training seed | FlyBrain adaptive | Same FlyBrain, fixed 1 step | A* + same world model |
|---|---:|---:|---:|
| 42 | 27/30 | 23/30 | 30/30 |
| 71 | 24/30 | 28/30 | 30/30 |
| 123 | 26/30 | 23/30 | 30/30 |
| **Total** | **77/90** | **74/90** | **90/90** |

The adaptive budget **does not consistently beat** fixed depth across seeds;
the overall three-task difference is not evidence of a robust causal improvement.
On these environments, the existing planner remains better. Longer term,
use matched compute budgets and independent tasks with many more seeds.

The included `examples/pretrained_fly_v0.3.pt` is trained on seed 42 only.
It must not be described as validated across model seeds by its own checkpoint.

## Real vs imagined

- `decide(...)` returns `verification: not_executed`; it cannot declare PASS.
- `FlySession` lives per episode and can be reset explicitly.
- For each actual step, call `accept_outcome(action, success)` with a verified
  result, then observe the next real state. Never treat a model prediction as a
  verified outcome.
- API requests are stateless. `history` is *client-asserted*, bounded to 64
  steps and replayed to reconstruct recurrent state. It is **not authenticated
  execution evidence**. Do not use it for verified outcome claims.
- Inference on CPU, no network required beyond the optional loopback HTTP API.

## Out of scope

No autonomous online training, no multi-task LLM reasoning, no learned symbolic
logic, no real tool execution, no private controllers and no automatic access to
local workspaces. These require future independent evidence.
