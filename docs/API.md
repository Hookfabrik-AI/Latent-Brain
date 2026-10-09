# HTTP API · v0.2 (experimental)

A small, read-only CPU inference API. Uses only Python's standard library for the HTTP server and the existing PyTorch model. No external LLM API, online training, or access tokens are involved.

## Start

```bash
python -m pip install -e '.[test]'
python -m latent_brain serve --checkpoint examples/pretrained_v0.1.pt --host 127.0.0.1 --port 8767
```

The model is loaded once. Startup fails if the checkpoint is missing or incompatible.

## Endpoints

- `GET /v1/health` returns `{ "status": "ok", "device": "cpu", "model_loaded": true, "api_version": "v1", "model": { ... } }`.
- `POST /v1/predict` accepts `{ "rows": ["...", ".#.", "..."], "queries": [{ "state": [0,0], "action": "right" }] }`. Each result provides `move_probability` (threshold 0.5) and `predicted_can_move`. Action names: `right`, `down`, `left`, `up`; integers 0..3 are also supported.
- `POST /v1/plan` accepts `{ "rows": ["...", ".#.", "..."], "max_nodes": 5000 }`. Returns `actions`/`action_ids` or `null` and **always** `"verification": "not_executed"`. The planner uses A* and learned transition predictions; it does **not** replay the result through the true environment.

Rows must be 2..25 high, 2..25 wide, rectangular, use only `.` and `#`, and leave top-left/bottom-right cells open. At most 256 predictions and 64 KiB of JSON are accepted per request. Invalid inputs yield JSON error responses with HTTP 400, 413, 415 or 404.

## Security & limitations

The API binds **only to loopback**, because it has no authentication, rate limiting, or network hardening. Do not expose it through a reverse proxy or port forward without adding authentication and safeguards. No CORS is enabled, and inference does not mutate the trained weights.

A planner's candidate path is **not a verification claim**. The checkpoint remains the v0.1 world model; adding HTTP endpoints does not improve its reasoning accuracy. LLM integrations, multi-step latent state learning and online adaptation are future work, not capabilities of this API.


## `POST /v1/fly/decide` (optional experimental extension)

Requires the server to be started with both `--checkpoint <transition.pt>` and
`--fly-checkpoint <fly.pt>`. Without the latter, returns `503 fly_model_not_loaded`.

```json
{
  "rows": ["...", ".#.", "..."],
  "state": [0, 0],
  "history": [{"state": [0, 0], "action": "right", "success": true}],
  "forced_budget": 2
}
```

Only `rows` and `state` are required. `history` contains up to 64 previous
client-asserted observations/outcomes; these reconstruct recurrent hidden state
per request without storing cross-user sessions. `forced_budget` is an optional
controlled ablation (1, 2 or 4). Returns chosen `action`, `action_id`, learned
policy and budget probabilities, chosen `thinking_budget`, number of
hypothetical expansions and **`verification: not_executed`**. This endpoint
never performs actions, accesses external tools or asserts success.

For repeatable experiments, request the same history each time. An authenticated,
server-side session system is **not** provided. Do not expose the localhost-only
HTTP server publicly.
