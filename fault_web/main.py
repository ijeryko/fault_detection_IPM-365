import os
import time
from datetime import datetime
from typing import Any, Dict

import eventlet
eventlet.monkey_patch()

from dotenv import load_dotenv
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from services.mongo import (
    get_mongo,
    ensure_indexes,
    upsert_device_last_seen,
    insert_event,
    get_enabled_alert_rules,
    log_notification,
    last_sent_in_cooldown,
    get_latest_new_result_events,
)
from services.queue import EmailQueue, EmailJob
from services.mailer import GmailMailer
from services.alerts import should_trigger_rule, build_email

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "secret")

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Keep original thesis-simple device auth
DEVICE_TOKENS = {
    "pi-01": "device-secret-token-CHANGE-ME",
}

def now_ms() -> int:
    return int(time.time() * 1000)

def authorized(payload: Dict[str, Any]) -> bool:
    device = payload.get("device_id")
    token = payload.get("token")
    return bool(device and token and DEVICE_TOKENS.get(device) == token)

def parse_event_ts(payload: Dict[str, Any]) -> datetime:
    ts = payload.get("timestamp")
    if ts is None:
        return datetime.utcnow()
    if isinstance(ts, (int, float)):
        if ts > 10_000_000_000:
            return datetime.utcfromtimestamp(ts / 1000.0)
        return datetime.utcfromtimestamp(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return datetime.utcnow()
    return datetime.utcnow()

# Mongo Atlas
mongo_client, db = get_mongo()
if db is not None:
    ensure_indexes(db)
else:
    app.logger.warning("MongoDB disabled...")

# Email
mailer = GmailMailer(
    gmail_address=os.getenv("GMAIL_ADDRESS", ""),
    app_password=os.getenv("GMAIL_APP_PASSWORD", ""),
    from_name=os.getenv("GMAIL_FROM_NAME", "Fault Monitor"),
)
default_recipients = [s.strip() for s in os.getenv("ALERT_DEFAULT_RECIPIENTS", "").split(",") if s.strip()]

email_queue = EmailQueue(enabled=mailer.is_configured(), logger=app.logger)
email_queue.start()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    if mongo_client is None or db is None:
        return {"status": "ready", "db": "disabled"}
    try:
        mongo_client.admin.command("ping")
        return {"status": "ready", "db": "ok"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}, 503

@socketio.on("connect")
def connect():
    emit("server_hello", {"message": "connected"})

    if db is not None:
        try:
            events = get_latest_new_result_events(db, limit=100)

            # Send oldest -> newest so UI fills chronologically
            events.reverse()

            for e in events:
                emit("new_result", e)  # triggers your existing dashboard.js handler
        except Exception as ex:
            app.logger.exception(f"Failed to replay latest events: {ex}")

@socketio.on("device_batch")
def device_batch(payload: Dict[str, Any]):
    if not authorized(payload):
        emit("device_ack", {"ok": False})
        return

    payload["timestamp"] = payload.get("timestamp") or now_ms()

    inserted_id = None
    if db is not None:
        try:
            device_id = payload.get("device_id")
            upsert_device_last_seen(db, device_id)
            inserted_id = insert_event(db, payload, parse_event_ts(payload))
        except Exception as e:
            app.logger.exception(f"DB insert failed: {e}")

    # Keep same emit as original repo
    socketio.emit(
        "new_result",
        {"prediction": payload.get("prediction"), "timestamp": payload.get("timestamp")},
    )

    # Alerts + email (async)
    # if db is not None and mailer.is_configured():
    #     try:
    #         device_id = payload.get("device_id")
    #         pred = payload.get("prediction") or {}
    #         label = pred.get("label", "unknown")

    #         rules = get_enabled_alert_rules(db, device_id)
    #         for rule in rules:
    #             if not should_trigger_rule(rule, payload):
    #                 continue

    #             cooldown = int(rule.get("conditions", {}).get("cooldown_minutes", rule.get("cooldown_minutes", 0)) or 0)
    #             if cooldown > 0 and last_sent_in_cooldown(db, rule["_id"], device_id, label, cooldown):
    #                 continue

    #             recipients = rule.get("recipients") or default_recipients
    #             if not recipients:
    #                 continue

    #             subject, body = build_email(rule, payload)

    #             def job_fn(rule_id=rule["_id"], event_id=inserted_id, recips=recipients, lbl=label):
    #                 try:
    #                     mailer.send_email(recips, subject, body)
    #                     log_notification(db, rule_id, device_id, event_id, lbl, recips, "sent", None)
    #                 except Exception as ex:
    #                     log_notification(db, rule_id, device_id, event_id, lbl, recips, "failed", str(ex))

    #             email_queue.enqueue(EmailJob(fn=job_fn))
    #     except Exception as e:
    #         app.logger.exception(f"Alert eval failed: {e}")

    emit("device_ack", {"ok": True})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("APP_ENV", "development").lower() == "development"
    socketio.run(app, host="0.0.0.0", port=port, debug=debug)