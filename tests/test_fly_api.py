"""Loopback FlyBrain API has no mutable shared hidden state or false PASS."""
import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from latent_brain.api import BrainService, create_server
from latent_brain.brain import LatentBrain
from latent_brain.fly import FlyBrain


@pytest.fixture(scope="module")
def endpoint():
    service = BrainService(LatentBrain(16), FlyBrain(16))
    server = create_server(service, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def invoke(payload):
        request = Request(f"http://127.0.0.1:{server.server_port}/v1/fly/decide",
                          data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=4) as result:
                return result.status, json.load(result)
        except HTTPError as error:
            return error.code, json.load(error)
    yield invoke
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def test_fly_endpoint_returns_unverified_action_and_telemetry(endpoint):
    inp = {"rows": ["...", ".#.", "..."], "state": [0, 0], "forced_budget": 2}
    status, result = endpoint(inp)
    assert status == 200
    assert result["thinking_budget"] == 2
    assert result["verification"] == "not_executed"
    assert result["action_id"] is None or 0 <= result["action_id"] < 4
    assert endpoint(inp) == (status, result)  # No cross-request state leakage


def test_fly_endpoint_accepts_bounded_history(endpoint):
    status, data = endpoint({"rows": ["..", ".."], "state": [1, 0],
                             "history": [{"state": [0, 0], "action": "right", "success": True}]})
    assert status == 200 and data["verification"] == "not_executed"


@pytest.mark.parametrize("payload", [
    {"rows": ["..", ".."], "state": [True, 0]},
    {"rows": ["..", ".."], "state": [0, 0], "forced_budget": 3},
    {"rows": ["..", ".."], "state": [0, 0], "history": ["wrong"]},
    {"rows": ["..", ".."], "state": [0, 0], "history": [{"state": [0, 0], "action": 1, "success": 1}]},
    {"rows": ["..", ".."], "state": [0, 0], "history": [{}] * 65},
])
def test_fly_endpoint_rejects_invalid_or_oversized_memory(endpoint, payload):
    status, data = endpoint(payload)
    assert status == 400 and data["error"] == "invalid_request"
