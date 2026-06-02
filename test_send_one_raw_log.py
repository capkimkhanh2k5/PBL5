import sys
import random
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore


SERVICE_ACCOUNT_PATH = "backend/src/main/resources/serviceAccountKey.json"


def init_firestore():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred)

    return firestore.client()


def send_one_raw_log(bin_id: str):
    db = init_firestore()

    payload = {
        "fillOrganic": random.randint(0, 40),
        "fillRecycle": random.randint(0, 40),
        "fillNonRecycle": random.randint(0, 40),
        "fillHazardous": random.randint(0, 40),

        # QUAN TRỌNG: backend listener đang đọc field này
        "recordedAt": firestore.SERVER_TIMESTAMP,

        # Không bắt buộc, thêm để dễ nhìn khi debug
        "source": "manual_test_script",
    }

    doc_ref = (
        db.collection("bin_raw_sensor_logs")
        .document(bin_id)
        .collection("logs")
        .document()
    )

    doc_ref.set(payload)

    print(f"✅ Sent one raw sensor log for {bin_id}")
    print(f"📌 Path: bin_raw_sensor_logs/{bin_id}/logs/{doc_ref.id}")
    print(f"🕒 Local time: {datetime.now()}")


if __name__ == "__main__":
    bin_id = sys.argv[1] if len(sys.argv) > 1 else "bin_001"
    send_one_raw_log(bin_id)