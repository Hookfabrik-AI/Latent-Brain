"""Deterministic API contract and real loopback HTTP tests."""
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import torch

from latent_brain.api import BrainService, RequestError, create_server, parse_maze
from latent_brain.brain import LatentBrain
from latent_brain.world import features, Maze


@pytest.fixture(scope="module")
def client():
    torch.manual_seed(2001)
    service = BrainService(LatentBrain(hidden_size=16, thinking_steps=2))
    server = create_server(service, port=0)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()

    def request(method, path, payload=None, content_type="application/json"):
        body = json.dumps(payload).encode() if payload is not None else None
        req = Request(f"http://127.0.0.1:{server.server_port}{path}", data=body, method=method)
        if body is not None:
            req.add_header("Content-Type", content_type)
        try:
            with urlopen(req, timeout=3) as resp:
                return resp.status, json.loads(resp.read())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read())

    yield service, request
    server.shutdown()
    server.server_close()
    worker.join(timeout=3)


def test_health_is_cpu_and_model_loaded(client):
    _, request = client
    status, value = request("GET", "/v1/health")
    assert status == 200 and value["device"] == "cpu"
    assert value["api_version"] == "v1" and value["model_loaded"] is True
    assert value["model"]["thinking_steps"] == 2


def test_prediction_matches_model_probabilities(client):
    service, request = client
    payload = {"rows": ["...", ".#.", "..."],
               "queries": [{"state": [0, 0], "action": "right"},
                           {"state": [1, 0], "action": 1}]}
    status, value = request("POST", "/v1/predict", payload)
    assert status == 200 and len(value["predictions"]) == 2
    maze = Maze(tuple(payload["rows"]))
    xs = torch.tensor([features(maze, (0, 0), 0), features(maze, (1, 0), 1)])
    actual = service.brain.probability(xs).tolist()
    for output, p in zip(value["predictions"], actual):
        assert output["move_probability"] == pytest.approx(p, abs=1e-6)
        assert output["predicted_can_move"] == (p >= 0.5)


def test_plan_is_marked_unverified(client):
    _, request = client
    status, response = request("POST", "/v1/plan", {"rows": ["...", "...", "..."]})
    assert status == 200
    assert response["verification"] == "not_executed"
    assert response["planner"] == "a_star_with_learned_transitions"
    assert response["actions"] is None or all(action in ("up", "down", "left", "right") for action in response["actions"])


@pytest.mark.parametrize("rows", [[], ["."], ["...", ".."], ["#.", ".."], ["..", ".#"], ["xx", ".."], ["." * 26, "." * 26]])
def test_invalid_mazes_rejected(client, rows):
    _, request = client
    status, value = request("POST", "/v1/plan", {"rows": rows})
    assert status == 400 and value["error"] == "invalid_request"


def test_invalid_query_rejected(client):
    _, request = client
    rows = ["..", ".."]
    for queries in ([], [{"state": [True, 0], "action": 0}],
                    [{"state": [3, 0], "action": 0}],
                    [{"state": [0, 0], "action": "invalid"}],
                    [{"state": [0, 0], "action": False}],
                    [{"state": [0, 0], "action": "right"}] * 257):
        code, value = request("POST", "/v1/predict", {"rows": rows, "queries": queries})
        assert code == 400 and value["error"] == "invalid_request"


def test_rejects_bad_search_budget(client):
    _, request = client
    for nodes in (True, 0, -1, 5001, 1.1):
        status, _ = request("POST", "/v1/plan", {"rows": ["..", ".."], "max_nodes": nodes})
        assert status == 400


def test_api_errors_are_structured(client):
    _, request = client
    status, _ = request("GET", "/other")
    assert status == 404
    status, _ = request("POST", "/v1/predict", {"rows": ["..", ".."]}, "text/plain")
    assert status == 415
    status, _ = request("POST", "/v1/predict", ["not-an-object"])
    assert status == 400
    status, _ = request("POST", "/v1/plan", {"rows": ["..", ".."], "extra": "x" * 65536})
    assert status == 413


def test_non_loopback_cannot_be_enabled(client):
    service, _ = client
    with pytest.raises(ValueError, match="loopback"):
        create_server(service, "0.0.0.0", 0)


def test_parse_maze_no_asserts_for_bad_input():
    with pytest.raises(RequestError):
        parse_maze(["#.", ".."])
