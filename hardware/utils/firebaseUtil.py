"""
firebaseUtil.py  —  Firebase Utility  v2.0
==========================================
Cập nhật đồng thời:
  - Realtime Database : /bins/{bin_id}/...
  - Firestore         : collection bin_realtime_status / document {bin_id}

Thông tin bin_id, fill_levels, ... được truyền vào lúc gọi hàm
để thuận tiện cho việc lấy dữ liệu từ cảm biến / Arduino sau này.

Yêu cầu:
    pip install firebase-admin
    File firebase_credit.json phải nằm cùng thư mục (hoặc chỉnh FIREBASE_CRED_PATH).
"""

import os
import sys
import time
import threading
from typing import Optional

import firebase_admin
from firebase_admin import credentials, db as rtdb, firestore

from dotenv import load_dotenv
# Load biến môi trường từ file .env (nếu có)
load_dotenv()

# ============================================================
# CẤU HÌNH FIREBASE
# ============================================================

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)

FIREBASE_CRED_PATH = os.path.join(_CURR_DIR, os.getenv("FIREBASE_CRED_PATH"))
FIREBASE_DB_URL    = os.getenv("FIREBASE_DB_URL")

# Map ngăn rác → field trong Realtime DB
BIN_TO_RTDB_FIELD = {
    "ORGANIC":    "fill_organic",
    "RECYCLABLE": "fill_recycle",
    "HAZARDOUS":  "fill_hazardous",
    "OTHER":      "fill_non_recycle",
}

# Map ngăn rác → field trong Firestore (camelCase theo ảnh Firestore)
BIN_TO_FIRESTORE_FIELD = {
    "ORGANIC":    "fillOrganic",
    "RECYCLABLE": "fillRecycle",
    "HAZARDOUS":  "fillHazardous",
    "OTHER":      "fillNonRecycle",
}


# ============================================================
# KHỞI TẠO
# ============================================================

_firebase_app = None
_firestore_client = None


def init_firebase() -> bool:
    """
    Khởi tạo Firebase Admin SDK (Realtime DB + Firestore).
    Trả về True nếu thành công, False nếu thất bại.
    """
    global _firebase_app, _firestore_client

    if not os.path.exists(FIREBASE_CRED_PATH):
        print(f"[WARN] Firebase credential không tìm thấy: {FIREBASE_CRED_PATH}")
        print("[WARN] Firebase sẽ bị tắt — chỉ chạy local.")
        return False

    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        _firebase_app = firebase_admin.initialize_app(
            cred, {'databaseURL': FIREBASE_DB_URL}
        )
        _firestore_client = firestore.client()
        print("[FIREBASE] Kết nối thành công (Realtime DB + Firestore).")
        return True
    except Exception as e:
        print(f"[FIREBASE ERROR] init: {e}")
        return False


# ============================================================
# CẬP NHẬT TRẠNG THÁI THÙNG RÁC
# ============================================================

def firebase_update_bin(
    firebase_ok: bool,
    bin_id: str,
    bin_type: str,
    locked_class: str,
    battery_level: Optional[int]   = None,
    temperature:   Optional[float] = None,
    fill_levels:   Optional[dict]  = None,
):
    """
    Tăng fill_counter tương ứng và ghi last_updated lên:
      1. Realtime Database : /bins/{bin_id}
      2. Firestore         : bin_realtime_status / {bin_id}

    Chạy trong thread riêng để không block main loop.

    Args:
        firebase_ok:   Kết quả từ init_firebase() — False nếu offline.
        bin_id:        ID thùng rác (vd: "bin_001").
        bin_type:      Ngăn rác (ORGANIC / RECYCLABLE / HAZARDOUS / OTHER).
        locked_class:  Class rác đã phân loại (vd: "Plastic").
        battery_level: % pin (None → dùng giá trị placeholder).
        temperature:   Nhiệt độ °C (None → dùng giá trị placeholder).
        fill_levels:   Dict từ arduinoUtil.read_fill_levels():
                       {"ORGANIC": {"distance_cm": 12.0, "fill_pct": 60.0}, ...}
                       Nếu None → không cập nhật fill_pct lên Firebase.
    """
    if not firebase_ok:
        print(f"[FIREBASE] Skip (offline): {bin_id} / {bin_type} / {locked_class}")
        return

    import random

    rtdb_field      = BIN_TO_RTDB_FIELD.get(bin_type, "fill_non_recycle")
    firestore_field = BIN_TO_FIRESTORE_FIELD.get(bin_type, "fillNonRecycle")
    bat             = battery_level if battery_level is not None else random.randint(20, 100)
    temp            = temperature   if temperature   is not None else round(random.uniform(20.0, 35.0), 1)
    now_ts          = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def _do_update():
        # ── 1. Realtime Database ──────────────────────────────────────
        try:
            ref     = rtdb.reference(f"/bins/{bin_id}")
            current = ref.child(rtdb_field).get() or 0

            rtdb_payload = {
                rtdb_field:      current + 1,
                "battery_level": bat,
                "last_updated":  now_ms,
                "status":        "ONLINE",
                "temperature":   temp,
            }

            # Nếu có dữ liệu cảm biến → ghi fill_pct từng ngăn
            if fill_levels:
                for b_type, info in fill_levels.items():
                    key      = BIN_TO_RTDB_FIELD.get(b_type)
                    fill_pct = info.get("fill_pct")
                    if key and fill_pct is not None:
                        rtdb_payload[f"{key}_pct"] = fill_pct

            ref.update(rtdb_payload)
            print(f"[FIREBASE][RTDB] Updated /bins/{bin_id}/{rtdb_field} = {current + 1}")

        except Exception as e:
            print(f"[FIREBASE ERROR][RTDB] {e}")

        # ── 2. Firestore: bin_realtime_status ─────────────────────────
        try:
            doc_ref = _firestore_client.collection("bin_realtime_status").document(bin_id)
            snap    = doc_ref.get()
            old_val = snap.to_dict().get(firestore_field, 0) if snap.exists else 0

            fs_payload = {
                "id":            bin_id,
                firestore_field: old_val + 1,
                "batteryLevel":  bat,
                "lastUpdated":   now_ts,
                "status":        "ONLINE",
                "temperature":   temp,
            }

            if fill_levels:
                for b_type, info in fill_levels.items():
                    fs_key   = BIN_TO_FIRESTORE_FIELD.get(b_type)
                    fill_pct = info.get("fill_pct")
                    if fs_key and fill_pct is not None:
                        fs_payload[f"{fs_key}Pct"] = fill_pct

            doc_ref.set(fs_payload, merge=True)
            print(f"[FIREBASE][FIRESTORE] Updated bin_realtime_status/{bin_id}/{firestore_field} = {old_val + 1}")

        except Exception as e:
            print(f"[FIREBASE ERROR][FIRESTORE] {e}")

    threading.Thread(target=_do_update, daemon=True).start()


# ============================================================
# ĐÁNH DẤU THÙNG RÁC ONLINE KHI KHỞI ĐỘNG
# ============================================================

def firebase_set_online(firebase_ok: bool, bin_id: str):
    """
    Đặt trạng thái bin thành ONLINE lên cả RTDB lẫn Firestore.
    Gọi một lần sau khi init_firebase() thành công.
    """
    if not firebase_ok:
        return

    #Get current timestamp as string
    now_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def _set():
        try:
            rtdb.reference(f"/bins/{bin_id}/status").set("ONLINE")
            rtdb.reference(f"/bins/{bin_id}/last_updated").set(now_ts)
            print(f"[FIREBASE][RTDB] {bin_id} → ONLINE")
        except Exception as e:
            print(f"[FIREBASE ERROR][RTDB] set_online: {e}")

        try:
            _firestore_client.collection("bin_realtime_status").document(bin_id).set(
                {"id": bin_id, "status": "ONLINE", "lastUpdated": now_ts},
                merge=True,
            )
            print(f"[FIREBASE][FIRESTORE] {bin_id} → ONLINE")
        except Exception as e:
            print(f"[FIREBASE ERROR][FIRESTORE] set_online: {e}")

    threading.Thread(target=_set, daemon=True).start()