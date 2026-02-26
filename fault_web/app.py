from __future__ import annotations
import time
from collections import deque
from typing import Any, Dict, Deque

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Device authentication
DEVICE_TOKENS = {
    "pi-01": "device-secret-token-CHANGE-ME",
}

MAX_EVENTS = 500

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"

socketio = SocketIO(app, cors_allowed_origins="*")

events: Deque[Dict[str, Any]] = deque(maxlen=MAX_EVENTS)


def now_ms():
    return int(time.time() * 1000)


def authorized(payload):
    device = payload.get("device_id")
    token = payload.get("token")
    if not device or not token:
        return False
    return DEVICE_TOKENS.get(device) == token


@app.route("/")

def home():
    # This will look for a file named home.html in your 'templates' folder
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@socketio.on("connect")
def connect():
    emit("server_hello", {"message": "connected"})


@socketio.on("device_batch")
def device_batch(payload):

    if not authorized(payload):
        emit("device_ack", {"ok": False})
        return

    # Ensure timestamp exists
    payload["timestamp"] = payload.get("timestamp") or now_ms()

    # Save event history
    events.append(payload)

    # --- FIXED EMIT: include the nested prediction object ---
    socketio.emit(
        "new_result",
        {
            "prediction": payload.get("prediction"),  # now matches dashboard.js
            "timestamp": payload.get("timestamp"),
        },
    )

    emit("device_ack", {"ok": True})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)