from typing import Any, Dict, Tuple

def should_trigger_rule(rule: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    pred = payload.get("prediction") or {}
    label = pred.get("label")
    confidence = float(pred.get("confidence", 0.0))

    conditions = rule.get("conditions", {})
    labels = conditions.get("labels") or rule.get("labels") or []
    min_conf = conditions.get("min_confidence", rule.get("min_confidence", 0.0))

    if labels and label not in labels:
        return False
    if confidence < float(min_conf):
        return False
    return True

def build_email(rule: Dict[str, Any], payload: Dict[str, Any]) -> Tuple[str, str]:
    device_id = payload.get("device_id", "unknown")
    pred = payload.get("prediction") or {}
    label = pred.get("label", "unknown")
    confidence = pred.get("confidence", 0.0)
    timestamp = payload.get("timestamp")

    subject = f"Fault Alert: {label} on {device_id}"
    body = (
        f"Fault detection alert\n\n"
        f"Device: {device_id}\n"
        f"Label: {label}\n"
        f"Confidence: {confidence}\n"
        f"Timestamp: {timestamp}\n"
    )
    return subject, body