"""Local decision service for the Unity live demo — the packaged "NPC brain".

    python -m experiments.rq3.live_server --config experiments/rq3/live_config_demo.json

Run from ``code/``. This is a localhost-only inter-process channel between the
Unity frontend and the Python decision logic: it never listens beyond
127.0.0.1 and involves no browser. In the released build it runs invisibly as
a PyInstaller subprocess that Unity starts and stops.

Endpoints (spec: docs/specs/2026-07-17-rq3-sequence-interface-design.md §5):

    GET  /state   world + events + unlock flags + NPC roster
    POST /step    one decision cycle for every NPC
    POST /event   {location, event, active} or {location, unlocked} -> new state
    POST /npc     {slot, ocean, reset_buffers?} edits a personality;
                  {name, ocean?, policy?, checkpoint?} appends an NPC -> new state
    POST /reset   authored world + startup roster restored -> new state

Errors: 400 with {"error": msg} for bad input, 404 for unknown paths.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .live_session import LiveSession


class LiveHandler(BaseHTTPRequestHandler):
    """One request handler class per server, bound to its LiveSession."""

    session: LiveSession                  # set by make_server

    # requests are sequential (single Unity client, single-threaded server);
    # the default per-request stderr line is noise next to the startup roster
    def log_message(self, fmt, *args):    # noqa: D102
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:             # noqa: N802 (stdlib naming)
        if self.path == "/state":
            self._send(200, self.session.state())
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:            # noqa: N802
        try:
            if self.path == "/step":
                self._send(200, self.session.step())
            elif self.path == "/event":
                body = self._read_body()
                if "unlocked" in body:
                    self.session.set_unlocked(body["location"], body["unlocked"])
                else:
                    self.session.set_event(body["location"], body["event"],
                                           body["active"])
                self._send(200, self.session.state())
            elif self.path == "/npc":
                body = self._read_body()
                if "slot" in body:
                    self.session.set_personality(
                        body["slot"], body.get("ocean", {}),
                        reset_buffers=bool(body.get("reset_buffers", False)))
                else:
                    self.session.add_npc(body)
                self._send(200, self.session.state())
            elif self.path == "/reset":
                self.session.reset()
                self._send(200, self.session.state())
            else:
                self._send(404, {"error": f"unknown path {self.path}"})
        except (ValueError, KeyError) as e:
            # covers malformed JSON (JSONDecodeError is a ValueError), missing
            # body keys, unknown location/event ids, and locked force targets
            self._send(400, {"error": str(e)})


def make_server(config_path: str | Path, port: int | None = None) -> HTTPServer:
    """Build the server; ``port=0`` binds an ephemeral port (tests)."""
    session = LiveSession.from_config(config_path)
    if port is None:
        port = int(session.config.get("port", 8973))
    handler = type("BoundLiveHandler", (LiveHandler,), {"session": session})
    return HTTPServer(("127.0.0.1", port), handler)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=None,
                    help="override the config's port")
    args = ap.parse_args()
    server = make_server(args.config, args.port)
    session = server.RequestHandlerClass.session
    host, port = server.server_address
    print(f"live demo brain on http://{host}:{port} "
          f"(world: {session.config['world']})")
    for npc in session.npcs:
        ocean = "  ".join(f"{k}={v:+.2f}"
                          for k, v in zip("OCEAN", npc.personality.vector))
        print(f"  [{npc.slot}] {npc.name:<14} {npc.policy_label:<10} {ocean}")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
