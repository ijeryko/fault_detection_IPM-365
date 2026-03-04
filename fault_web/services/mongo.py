import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from pymongo import MongoClient, ASCENDING, DESCENDING


def get_mongo() -> Tuple[Optional[MongoClient], Optional[Any]]:
    uri = os.getenv("MONGO_URI", "")
    dbname = os.getenv("MONGO_DB_NAME", "fault_web")
    if not uri:
        return None, None

    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=5000,
        retryWrites=True,
    )
    db = client[dbname]
    return client, db


def ensure_indexes(db):
    db.devices.create_index([("device_id", ASCENDING)], unique=True)
    db.devices.create_index([("last_seen_at", DESCENDING)])

    db.events.create_index([("device_id", ASCENDING), ("event_ts", DESCENDING)])
    db.events.create_index([("prediction.label", ASCENDING), ("event_ts", DESCENDING)])

    db.alert_rules.create_index([("enabled", ASCENDING), ("device_id", ASCENDING)])
    db.notification_logs.create_index([("rule_id", ASCENDING), ("sent_at", DESCENDING)])
    db.notification_logs.create_index([("device_id", ASCENDING), ("sent_at", DESCENDING)])
    db.notification_logs.create_index([("label", ASCENDING), ("sent_at", DESCENDING)])


def upsert_device_last_seen(db, device_id: str):
    db.devices.update_one(
        {"device_id": device_id},
        {
            "$set": {"last_seen_at": datetime.utcnow()},
            "$setOnInsert": {"device_id": device_id, "created_at": datetime.utcnow()},
        },
        upsert=True,
    )


def insert_event(db, payload: Dict[str, Any], event_ts: datetime):
    pred = payload.get("prediction") or {}
    doc = {
        "device_id": payload.get("device_id"),
        "event_ts": event_ts,
        "received_at": datetime.utcnow(),
        "prediction": pred,
        "metrics": payload.get("metrics"),
        "raw": payload,
    }
    return db.events.insert_one(doc).inserted_id


def get_enabled_alert_rules(db, device_id: str):
    return list(
        db.alert_rules.find(
            {"enabled": True, "$or": [{"device_id": device_id}, {"device_id": None}, {"device_id": {"$exists": False}}]}
        )
    )


def log_notification(db, rule_id, device_id: str, event_id, label: str, recipients: list[str], status: str, error: Optional[str]):
    db.notification_logs.insert_one(
        {
            "rule_id": rule_id,
            "device_id": device_id,
            "event_id": event_id,
            "label": label,
            "sent_to": recipients,
            "status": status,
            "error_message": error,
            "sent_at": datetime.utcnow(),
        }
    )


def last_sent_in_cooldown(db, rule_id, device_id: str, label: str, cooldown_minutes: int) -> bool:
    if cooldown_minutes <= 0:
        return False
    last = db.notification_logs.find_one(
        {"rule_id": rule_id, "device_id": device_id, "label": label, "status": "sent"},
        sort=[("sent_at", -1)],
    )
    if not last:
        return False
    last_sent = last.get("sent_at")
    if not last_sent:
        return False
    return (datetime.utcnow() - last_sent) < timedelta(minutes=cooldown_minutes)

def get_latest_new_result_events(db, limit: int = 100):
    cursor = db.events.find({}, sort=[("received_at", -1)], limit=limit)
    events = []
    for doc in cursor:
        raw = doc.get("raw") or {}
        pred = raw.get("prediction") or doc.get("prediction") or {}
        ts = raw.get("timestamp") or doc.get("event_ts")

        events.append({
            "prediction": {
                "label": pred.get("label", "Unknown"),
                "confidence": pred.get("confidence", 0),
            },
            "timestamp": ts,
        })

    # optional: oldest -> newest for nicer filling
    events.reverse()
    return events