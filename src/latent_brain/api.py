"""Local-only HTTP JSON interface to the CPU transition model, no web dependencies.

Predicted plans are candidate plans, NOT evidence of successful execution.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from typing import Any

import torch

from .planning import plan
from .world import ACTION_NAMES, Maze, features

MAX_REQUEST_BYTES = 65536
MAX_QUERIES = 256
MAX_SIDE = 25
MAX_SEARCH_NODES = 5000
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTION_NAMES)}


class RequestError(ValueError):
    """Malformed client input."""


def parse_maze(value: Any) -> Maze:
    if not isinstance(value, list) or not 2 <= len(value) <= MAX_SIDE:
        raise RequestError("rows must contain 2..25 strings")
    if not all(isinstance(row, str) for row in value):
        raise RequestError("each row must be a string")
    if not 2 <= len(value[0]) <= MAX_SIDE or any(len(row) != len(value[0]) for row in value):
        raise RequestError("rows must have the same width (2..25)")
    if any(set(row) - {".", "#"} for row in value):
        raise RequestError("rows may contain only '.' and '#'")
    if value[0][0] != "." or value[-1][-1] != ".":
        raise RequestError("start and goal must be open")
    return Maze(tuple(value))


def parse_action(value: Any) -> int:
    if isinstance(value, str) and value in ACTION_TO_ID:
        return ACTION_TO_ID[value]
    if type(value) is int and 0 <= value < len(ACTION_NAMES):
        return value
    raise RequestError("action must be right/down/left/up or integer 0..3")


def parse_state(value: Any, maze: Maze) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise RequestError("state must be [x, y]")
    x, y = value
    if type(x) is not int or type(y) is not int:
        raise RequestError("state coordinates must be integers")
    if not (0 <= x < maze.width and 0 <= y < maze.height):
        raise RequestError("state is outside the maze")
    return x, y


class BrainService:
    """Read-only inference; checkpoint is loaded once by the caller."""

    def __init__(self, brain: torch.nn.Module):
        self.brain = brain.cpu().eval()

    def health(self) -> dict:
        return {"status": "ok", "api_version": "v1", "device": "cpu",
                "model": self.brain.metadata(), "model_loaded": True}

    def predict(self, payload: dict) -> dict:
        maze = parse_maze(payload.get("rows"))
        queries = payload.get("queries")
        if not isinstance(queries, list) or not 1 <= len(queries) <= MAX_QUERIES:
            raise RequestError("queries must contain 1..256 items")
        inputs, parsed = [], []
        for query in queries:
            if not isinstance(query, dict):
                raise RequestError("each query must be an object")
            state = parse_state(query.get("state"), maze)
            action = parse_action(query.get("action"))
            parsed.append((state, action))
            inputs.append(features(maze, state, action))
        with torch.inference_mode():
            probabilities = self.brain.probability(torch.tensor(inputs, dtype=torch.float32)).tolist()
        return {
            "predictions": [
                {"state": list(state), "action": ACTION_NAMES[action],
                 "move_probability": round(float(p), 6), "predicted_can_move": p >= 0.5}
                for (state, action), p in zip(parsed, probabilities)
            ],
            "model_type": "learned_transition_predictor",
        }

    def proposed_plan(self, payload: dict) -> dict:
        maze = parse_maze(payload.get("rows"))
        max_nodes = payload.get("max_nodes", MAX_SEARCH_NODES)
        if type(max_nodes) is not int or not 1 <= max_nodes <= MAX_SEARCH_NODES:
            raise RequestError("max_nodes must be an integer from 1 to 5000")

        def predictor(m: Maze, queries):
            if not queries:
                return []
            xs = torch.tensor([features(m, s, a) for s, a in queries], dtype=torch.float32)
            with torch.inference_mode():
                return self.brain.probability(xs).ge(0.5).tolist()

        actions = plan(maze, predictor, max_nodes=max_nodes)
        return {
            "plan_found": actions is not None,
            "actions": [ACTION_NAMES[a] for a in actions] if actions is not None else None,
            "action_ids": actions,
            "planner": "a_star_with_learned_transitions",
            "verification": "not_executed",
            "notice": "Candidate plan is not evidence of successful real execution.",
        }


def create_server(service: BrainService, host: str = "127.0.0.1", port: int = 8767) -> ThreadingHTTPServer:
    """Create but do not start a server. Caller owns its lifecycle."""
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("API is loopback only; network exposure requires authentication")
    if not 0 <= port <= 65535:
        raise ValueError("invalid port")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send_json(self, status: int, data: dict) -> None:
            body = json.dumps(data, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/v1/health":
                return self.send_json(200, service.health())
            return self.send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path not in ("/v1/predict", "/v1/plan"):
                return self.send_json(404, {"error": "not_found"})
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                return self.send_json(415, {"error": "content_type_must_be_application_json"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.send_json(400, {"error": "invalid_content_length"})
            if length <= 0:
                return self.send_json(400, {"error": "empty_body"})
            if length > MAX_REQUEST_BYTES:
                self.close_connection = True
                return self.send_json(413, {"error": "request_too_large"})
            try:
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise RequestError("body must be a JSON object")
                result = service.predict(payload) if self.path == "/v1/predict" else service.proposed_plan(payload)
            except (RequestError, json.JSONDecodeError) as exc:
                return self.send_json(400, {"error": "invalid_request", "detail": str(exc)})
            except Exception:
                return self.send_json(500, {"error": "internal_error"})
            return self.send_json(200, result)

        def log_message(self, fmt: str, *args: Any) -> None:
            return  # No prompt / request body logging.

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def serve(checkpoint: str, host: str = "127.0.0.1", port: int = 8767) -> None:
    from .brain import load_checkpoint

    brain, _ = load_checkpoint(checkpoint)
    server = create_server(BrainService(brain), host, port)
    print(f"Latent Brain API: http://{host}:{server.server_port}/v1/health", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
