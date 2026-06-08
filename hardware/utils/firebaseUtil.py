"""
firebaseUtil.py  —  Firebase Utility  v3.0
==========================================
Cập nhật theo kiến trúc mới:
  - CHỈ dùng Firestore (không còn Realtime Database)
  - Ghi dữ liệu cảm biến vào:
      bin_raw_sensor_logs / {bin_id} / logs / {auto_id}
  - classification_logs vẫn được xử lý qua cloudinaryUtil

Cấu trúc document trong logs:
  fillOrganic      : number  (%)
  fillRecycle      : number  (%)
  fillNonRecycle   : number  (%)
  fillHazardous    : number  (%)
  recordedAt       : timestamp

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
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import SERVER_TIMESTAMP

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# CẤU HÌNH FIREBASE
# ============================================================

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)

FIREBASE_CRED_PATH = os.path.join(_CURR_DIR, os.getenv("FIREBASE_CRED_PATH", "firebase_credit.json"))
FIRESTORE_WRITE_TIMEOUT_SEC = float(os.getenv("FIRESTORE_WRITE_TIMEOUT_SEC", "4.0"))
FIRESTORE_COMMAND_TIMEOUT_SEC = float(os.getenv("FIRESTORE_COMMAND_TIMEOUT_SEC", "4.0"))

# Map ngăn rác → field fill_pct trong fill_levels dict (từ arduinoUtil)
BIN_TO_FILL_FIELD = {
    "ORGANIC":    "fillOrganic",
    "RECYCLABLE": "fillRecycle",
    "HAZARDOUS":  "fillHazardous",
    "OTHER":      "fillNonRecycle",
}

# ============================================================
# KHỞI TẠO
# ============================================================

_firebase_app     = None
_firestore_client = None


def init_firebase() -> bool:
    """
    Khởi tạo Firebase Admin SDK (Firestore).
    Trả về True nếu thành công, False nếu thất bại.
    """
    global _firebase_app, _firestore_client

    if not os.path.exists(FIREBASE_CRED_PATH):
        print(f"[WARN] Firebase credential không tìm thấy: {FIREBASE_CRED_PATH}")
        print("[WARN] Firebase sẽ bị tắt — chỉ chạy local.")
        return False

    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        _firebase_app = firebase_admin.initialize_app(cred)
        _firestore_client = firestore.client()
        print("[FIREBASE] Kết nối thành công (Firestore).")
        return True
    except Exception as e:
        print(f"[FIREBASE ERROR] init: {e}")
        return False


def get_firestore_client():
    """Trả về Firestore client (dùng chung với cloudinaryUtil nếu cần)."""
    return _firestore_client


# ============================================================
# ĐIỀU KHIỂN ON/OFF PHÂN LOẠI
# ============================================================

def firebase_get_classification_command(firebase_ok: bool, bin_id: str) -> Optional[dict]:
    """Đọc lệnh mới nhất tại bin_commands/{bin_id}; trả None nếu offline/không có."""
    if not firebase_ok or _firestore_client is None:
        return None

    try:
        doc = (
            _firestore_client
            .collection("bin_commands")
            .document(bin_id)
            .get(timeout=FIRESTORE_COMMAND_TIMEOUT_SEC)
        )
        if not doc.exists:
            return None
        data = doc.to_dict() or {}
        data["_id"] = doc.id
        return data
    except Exception as e:
        print(f"[FIREBASE ERROR][FIRESTORE] get_classification_command: {e}")
        return None


def firebase_update_classification_enabled(
    firebase_ok: bool,
    bin_id: str,
    enabled: bool,
) -> bool:
    """Cập nhật trạng thái thực tế vào bins_metadata/{bin_id}."""
    if not firebase_ok or _firestore_client is None:
        print(f"[FIREBASE] Skip metadata classification_enabled={enabled} (offline).")
        return False

    try:
        _firestore_client.collection("bins_metadata").document(bin_id).set(
            {
                "classification_enabled": bool(enabled),
                "classification_updated_at": SERVER_TIMESTAMP,
            },
            merge=True,
            timeout=FIRESTORE_COMMAND_TIMEOUT_SEC,
        )
        print(f"[FIRESTORE] bins_metadata/{bin_id}.classification_enabled={enabled}")
        return True
    except Exception as e:
        print(f"[FIREBASE ERROR][FIRESTORE] update_classification_enabled: {e}")
        return False


def firebase_update_command_status(
    firebase_ok: bool,
    bin_id: str,
    command_id: str,
    status: str,
    error_message: Optional[str] = None,
) -> bool:
    """Cập nhật trạng thái xử lý lệnh tại bin_commands/{bin_id}."""
    if not firebase_ok or _firestore_client is None:
        print(f"[FIREBASE] Skip command status={status} (offline).")
        return False

    payload = {
        "command_id": command_id,
        "status": status,
        "handled_at": SERVER_TIMESTAMP,
        "error_message": error_message,
    }

    try:
        _firestore_client.collection("bin_commands").document(bin_id).set(
            payload,
            merge=True,
            timeout=FIRESTORE_COMMAND_TIMEOUT_SEC,
        )
        print(f"[FIRESTORE] bin_commands/{bin_id}: {command_id} → {status}")
        return True
    except Exception as e:
        print(f"[FIREBASE ERROR][FIRESTORE] update_command_status: {e}")
        return False


def firebase_get_classification_enabled(
    firebase_ok: bool,
    bin_id: str,
    default: bool = True,
) -> bool:
    """Đọc trạng thái phân loại thực tế từ bins_metadata/{bin_id}; lỗi/offline thì dùng default."""
    if not firebase_ok or _firestore_client is None:
        return default

    try:
        doc = (
            _firestore_client
            .collection("bins_metadata")
            .document(bin_id)
            .get(timeout=FIRESTORE_COMMAND_TIMEOUT_SEC)
        )
        if not doc.exists:
            return default
        data = doc.to_dict() or {}
        value = data.get("classification_enabled")
        if isinstance(value, bool):
            return value
        return default
    except Exception as e:
        print(f"[FIREBASE ERROR][FIRESTORE] get_classification_enabled: {e}")
        return default


# ============================================================
# GHI LOG CẢM BIẾN VÀO bin_raw_sensor_logs
# ============================================================

def firebase_log_sensor(
    firebase_ok: bool,
    bin_id: str,
    fill_levels: Optional[dict] = None,
):
    """
    Ghi một bản ghi cảm biến vào:
        bin_raw_sensor_logs / {bin_id} / logs / {auto_id}

    Chạy trong thread riêng để không block main loop.

    Args:
        firebase_ok:  Kết quả từ init_firebase() — False nếu offline.
        bin_id:       ID thùng rác (vd: "bin_001").
        fill_levels:  Dict từ arduinoUtil.read_fill_levels() hoặc simulated:
                      {"ORGANIC": {"distance_cm": 12.0, "fill_pct": 60.0}, ...}
                      Nếu None → ghi 0.0 cho tất cả các ngăn.
    """
    if not firebase_ok:
        print(f"[FIREBASE] Skip (offline): sensor log cho {bin_id}")
        return

    def _do_log():
        try:
            # Lấy fill_pct từng ngăn (mặc định 0.0 nếu không có dữ liệu)
            def _get_pct(bin_type: str) -> float:
                if fill_levels and bin_type in fill_levels:
                    pct = fill_levels[bin_type].get("fill_pct")
                    if pct is not None:
                        return float(pct)
                return 0.0

            payload = {
                "fillOrganic":    _get_pct("ORGANIC"),
                "fillRecycle":    _get_pct("RECYCLABLE"),
                "fillNonRecycle": _get_pct("OTHER"),
                "fillHazardous":  _get_pct("HAZARDOUS"),
                "recordedAt":     SERVER_TIMESTAMP,
            }

            # Ghi vào sub-collection: bin_raw_sensor_logs/{bin_id}/logs
            logs_ref = (
                _firestore_client
                .collection("bin_raw_sensor_logs")
                .document(bin_id)
                .collection("logs")
            )
            doc_ref = logs_ref.add(payload, timeout=FIRESTORE_WRITE_TIMEOUT_SEC)
            print(f"[FIREBASE][FIRESTORE] Ghi sensor log: bin_raw_sensor_logs/{bin_id}/logs/{doc_ref[1].id}")

        except Exception as e:
            print(f"[FIREBASE ERROR][FIRESTORE] firebase_log_sensor: {e}")

    threading.Thread(target=_do_log, daemon=True).start()
