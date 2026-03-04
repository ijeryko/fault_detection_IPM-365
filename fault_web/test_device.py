import os
import time
import socketio
import random

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:5050")

DEVICE_ID = "pi-01"
TOKEN = "device-secret-token-CHANGE-ME"

sio = socketio.Client(logger=True, engineio_logger=False)

@sio.event
def connect():
    print("Connected to server")
    send_one()

@sio.on("device_ack")
def on_device_ack(data):
    print("device_ack:", data)

@sio.event
def disconnect():
    print("Disconnected")

def send_one():
    labels = ["healthy", "belt_fault", "off", "bearing_fault"]
    label = random.choice(labels)
    confidence = round(random.uniform(0.70, 0.99), 2)

    payload = {
        "device_id": DEVICE_ID,
        "token": TOKEN,
        "timestamp": int(time.time() * 1000),
        "prediction": {"label": label, "confidence": confidence},
        "metrics": {
            "vibration_rms": round(random.uniform(0.5, 3.0), 2),
            "temp_c": round(random.uniform(28.0, 60.0), 1),
        },
    }

    print("Sending payload:", payload)
    sio.emit("device_batch", payload)

def main():
    sio.connect(SERVER_URL, transports=["websocket"])
    try:
        while True:
            time.sleep(3)
            send_one()
    except KeyboardInterrupt:
        pass
    finally:
        sio.disconnect()

if __name__ == "__main__":
    main()