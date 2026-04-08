"""
firebaseUtil.py  —  Firebase Realtime Database Utility
=======================================================
Chứa toàn bộ cấu hình, khởi tạo và logic cập nhật Firebase
cho hệ thống smart trash bin.

Yêu cầu:
    pip install firebase-admin
    File firebase_credit.json phải nằm cùng thư mục (hoặc chỉnh FIREBASE_CRED_PATH).
"""

import os
import sys
import time
import random
import threading

import firebase_admin
from firebase_admin import credentials, db as firebase_db

# Thêm thư mục utils vào path để import
_CURR_DIR = os.path.join(os.path.dirname(__file__))
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)

# ============================================================
# CẤU HÌNH FIREBASE
# ============================================================
FIREBASE_CRED_PATH = "../keys/firebase_credit.json"
FIREBASE_DB_URL    = "https://pbl5-f21e6-default-rtdb.asia-southeast1.firebasedatabase.app/"
BIN_ID             = "bin_001"   # ← Đổi thành bin_002 / bin_003 nếu cần

# Map ngăn → Firebase field
BIN_TO_FIREBASE_FIELD = {
    "ORGANIC":    "fill_organic",
    "RECYCLABLE": "fill_recycle",
    "HAZARDOUS":  "fill_hazardous",
    "OTHER":      "fill_non_recycle",
}


# ============================================================
# KHỞI TẠO FIREBASE
# ============================================================

def init_firebase() -> bool | None:
    """
    Khởi tạo Firebase Admin SDK và kiểm tra kết nối.
    Trả về True nếu thành công, None nếu thất bại.
    """
    if not os.path.exists(FIREBASE_CRED_PATH):
        print(f"[WARN] Firebase credential không tìm thấy: {FIREBASE_CRED_PATH}")
        print("[WARN] Firebase sẽ bị tắt — chỉ chạy local.")
        return None
    try:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DB_URL})
        # Kiểm tra kết nối
        ref = firebase_db.reference(f"/bins/{BIN_ID}/status")
        ref.set("ONLINE")
        print(f"[FIREBASE] Kết nối thành công. Bin: {BIN_ID}")
        return True
    except Exception as e:
        print(f"[FIREBASE ERROR] {e}")
        return None


# ============================================================
# CẬP NHẬT FIREBASE
# ============================================================

def firebase_update_bin(firebase_ok, bin_type: str, locked_class: str):
    """
    Tăng fill_counter tương ứng và ghi last_updated.
    Chạy trong thread riêng để không block main loop.

    Args:
        firebase_ok:  Kết quả trả về từ init_firebase() — None nếu offline.
        bin_type:     Tên ngăn rác (ORGANIC / RECYCLABLE / HAZARDOUS / OTHER).
        locked_class: Tên class rác đã phân loại.
    """
    if not firebase_ok:
        print(f"[FIREBASE] Skip (offline): {bin_type} / {locked_class}")
        return

    field = BIN_TO_FIREBASE_FIELD.get(bin_type, "fill_non_recycle")

    def _do_update():
        try:
            ref = firebase_db.reference(f"/bins/{BIN_ID}")
            current = ref.child(field).get() or 0
            ref.update({
                field:           current + 1,
                "battery_level": random.randint(20, 100),   # ← Giả lập giá trị pin
                "last_updated":  int(time.time() * 1000),
                "status":        "ONLINE",
                "temperature":   round(random.uniform(20.0, 35.0), 1),  # ← Giả lập nhiệt độ
            })
            print(f"[FIREBASE] Updated {BIN_ID}/{field} = {current + 1}")
        except Exception as e:
            print(f"[FIREBASE ERROR] update failed: {e}")

    threading.Thread(target=_do_update, daemon=True).start()