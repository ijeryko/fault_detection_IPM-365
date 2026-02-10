from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Deque, Dict

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

# -----------------------
# Config
# -----------------------
DEVICE_TOKENS = {
    # must match sender's DEVICE_TOKEN for each device_id
    "pi-01": "device-secret-token-CHANGE-ME",
}

MAX_EVENTS = 500  # keep last N lines in memory


# -----------------------
# App setup
# -----------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")

socketio = SocketIO(
    app,
    cors_allowed_origins="*",   # tighten later (set your domain)
    async_mode="eventlet",
)

events: Deque[Dict[str, Any]] = deque(maxlen=MAX_EVENTS)


# -----------------------
# Helpers
# -----------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def format_event_line(e: Dict[str, Any]) -> str:
    pred = e.get("prediction", {}) or {}
    label = str(pred.get("label", "UNKNOWN")).upper()
    conf = float(pred.get("confidence", 0.0))
    rec = float(e.get("rec_duration_s", 0.0))
    return f"Result: {label} ({conf:.1f}%) | Rec Time: {rec:.2f}s"


def is_authorized_device(payload: Dict[str, Any]) -> bool:
    device_id = payload.get("device_id")
    token = payload.get("token")
    if not device_id or not token:
        return False
    return DEVICE_TOKENS.get(device_id) == token


# -----------------------
# Routes
# -----------------------
@app.get("/")
def root():
    return render_template("dashboard.html")


@app.get("/dashboard")
def dashboard():
    # Provide last events so the page isn't empty on load
    snapshot = list(events)
    snapshot_lines = [format_event_line(e) for e in snapshot]
    return render_template("dashboard.html", initial_lines=snapshot_lines)


# -----------------------
# Socket.IO handlers
# -----------------------
@socketio.on("connect")
def on_connect():
    # Browser connects here (not the device script)
    emit("server_hello", {"ts_ms": _now_ms(), "remote": request.remote_addr})


@socketio.on("device_batch")
def on_device_batch(payload: Dict[str, Any]):
    # This is what your sender emits: SOCKET_EVENT_NAME = "device_batch"
    if not isinstance(payload, dict) or not is_authorized_device(payload):
        emit("device_ack", {"ok": False, "reason": "unauthorized"})
        return

    # Drop raw samples for dashboard (keeps memory small)
    payload = dict(payload)
    payload.pop("samples", None)

    payload["ts_ms"] = _now_ms()
    events.append(payload)

    # Broadcast to all dashboards
    socketio.emit(
        "new_result",
        {
            "ts_ms": payload["ts_ms"],
            "device_id": payload.get("device_id"),
            "line": format_event_line(payload),
            "prediction": payload.get("prediction", {}),
            "rec_duration_s": payload.get("rec_duration_s"),
            "sps_est": payload.get("sps_est"),
        },
    )

    emit("device_ack", {"ok": True})


if __name__ == "__main__":
    # Listen on all interfaces so your Pi can connect to it
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
