"""Live HTTP layer: every endpoint + error paths over a real socket
(plan: docs/plans/2026-07-18-live-demo-mode.md)."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from experiments.rq3.live_server import make_server

SERVER_WORLD = {
    "locations": [
        {"id": "tavern",
         "features": {"social": 0.9, "stimulation": 0.7},
         "actions": [{"id": "chat", "features": {"social": 0.9}},
                     {"id": "drink", "features": {"stimulation": 0.6}}],
         "events": [
             {"name": "celebration", "active": False,
              "buff": {"social": 0.1}, "force_npc": None},
             {"name": "brawl_night", "active": False,
              "buff": {}, "force_npc": "tavern/drink"},
         ]},
        {"id": "cave",
         "features": {"exploration": 0.9, "risk": 0.7},
         "unlocked": False,
         "actions": [{"id": "spelunk",
                      "features": {"exploration": 0.9, "risk": 0.6}}],
         "events": [
             {"name": "expedition", "active": False,
              "buff": {}, "force_npc": "cave/spelunk"},
         ]},
    ]
}

PERSONALITIES = {"personalities": [
    {"name": "Alice", "ocean": {"extraversion": 1.0}},
]}


@pytest.fixture
def base_url(tmp_path):
    (tmp_path / "world.json").write_text(json.dumps(SERVER_WORLD), encoding="utf-8")
    (tmp_path / "personalities.json").write_text(
        json.dumps(PERSONALITIES), encoding="utf-8")
    cfg = {
        "world": str(tmp_path / "world.json"),
        "personalities": str(tmp_path / "personalities.json"),
        "port": 0, "mode": "sample", "seed": 7,
        "npcs": [{"name": "Alice", "policy": "scorer"}],
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    server = make_server(cfg_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


def get(url):
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post(url, payload=None, raw: bytes | None = None):
    data = raw if raw is not None else json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_state_roundtrip(base_url):
    status, state = get(base_url + "/state")
    assert status == 200
    assert state["step_count"] == 0
    assert [n["name"] for n in state["npcs"]] == ["Alice"]
    assert {l["id"] for l in state["locations"]} == {"tavern", "cave"}


def test_step_advances_and_has_step_shape(base_url):
    status, first = post(base_url + "/step")
    assert status == 200
    assert first["step_count"] == 1
    (entry,) = first["steps"]
    assert entry["name"] == "Alice"
    assert entry["moved"] is True
    assert entry["overridden"] is False
    assert {p["id"] for p in entry["location_probs"]} == {"tavern"}  # cave locked

    _, second = post(base_url + "/step")
    assert second["step_count"] == 2


def test_event_toggle_and_unlock_roundtrip(base_url):
    status, state = post(base_url + "/event",
                         {"location": "tavern", "event": "celebration",
                          "active": True})
    assert status == 200
    tavern = next(l for l in state["locations"] if l["id"] == "tavern")
    assert next(e for e in tavern["events"]
                if e["name"] == "celebration")["active"] is True

    status, state = post(base_url + "/event",
                         {"location": "cave", "unlocked": True})
    assert status == 200
    assert next(l for l in state["locations"] if l["id"] == "cave")["unlocked"] is True

    _, step = post(base_url + "/step")
    assert {p["id"] for p in step["steps"][0]["location_probs"]} == {"tavern", "cave"}


def test_force_event_via_http(base_url):
    post(base_url + "/event",
         {"location": "tavern", "event": "brawl_night", "active": True})
    _, step = post(base_url + "/step")
    (entry,) = step["steps"]
    assert (entry["location"], entry["action"]) == ("tavern", "drink")
    assert entry["overridden"] is True
    assert entry["location_probs"] == [] and entry["action_probs"] == []


def test_reset_restores_authored_state(base_url):
    post(base_url + "/step")
    post(base_url + "/event",
         {"location": "cave", "unlocked": True})
    status, state = post(base_url + "/reset")
    assert status == 200
    assert state["step_count"] == 0
    assert next(l for l in state["locations"] if l["id"] == "cave")["unlocked"] is False
    assert state["npcs"][0]["location"] == ""


def test_npc_edit_endpoint(base_url):
    status, state = post(base_url + "/npc",
                         {"slot": 0, "ocean": {"extraversion": -1.0}})
    assert status == 200
    assert state["npcs"][0]["ocean"] == [0.0, 0.0, -1.0, 0.0, 0.0]

    status, body = post(base_url + "/npc",
                        {"slot": 5, "ocean": {"extraversion": 0.0}})
    assert status == 400 and "slot" in body["error"]

    status, body = post(base_url + "/npc", {"slot": 0})   # no ocean dict
    assert status == 400 and "ocean" in body["error"]


def test_npc_add_endpoint_and_reset_discards(base_url):
    status, state = post(base_url + "/npc",
                         {"name": "Zed", "ocean": {"openness": 1.0}})
    assert status == 200
    assert [n["name"] for n in state["npcs"]] == ["Alice", "Zed"]

    _, state = post(base_url + "/reset")
    assert [n["name"] for n in state["npcs"]] == ["Alice"]


def test_error_paths(base_url):
    status, body = get(base_url + "/nope")
    assert status == 404 and "error" in body

    status, body = post(base_url + "/event",
                        {"location": "atlantis", "unlocked": True})
    assert status == 400 and "atlantis" in body["error"]

    status, body = post(base_url + "/event",
                        {"location": "tavern", "event": "eclipse", "active": True})
    assert status == 400 and "eclipse" in body["error"]

    status, body = post(base_url + "/event", raw=b"this is not json")
    assert status == 400 and "error" in body

    # force target locked -> the step itself reports the problem
    post(base_url + "/event",
         {"location": "cave", "event": "expedition", "active": True})
    status, body = post(base_url + "/step")
    assert status == 400 and "cave" in body["error"]
