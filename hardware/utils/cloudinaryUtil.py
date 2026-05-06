import os
import sys
import time
import uuid
import threading
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

import cv2

# ============================================================
# CẤU HÌNH
# ============================================================

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))

CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY    = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
BIN_ID = os.getenv("BIN_ID", "bin_001")  # ID thùng rác mặc định nếu không có trong .env

SAVE_ROOT_DIR = "PBL5"

# Thư mục lưu ảnh tạm trên local trước khi upload
LOCAL_SNAPSHOT_DIR = os.path.join(_CURR_DIR, "snapshots")

# Map ngăn rác → tên folder Cloudinary
BIN_TO_CLOUDINARY_FOLDER = {
    "ORGANIC":    "Biological",
    "RECYCLABLE": "Recyclable",
    "HAZARDOUS":  "Hazardous",
    "OTHER":      "General",
}

# ============================================================
# KHỞI TẠO CLOUDINARY
# ============================================================

_cloudinary_ok = False


def init_cloudinary() -> bool:
    """
    Đọc credentials từ biến môi trường (.env) và cấu hình Cloudinary SDK.
    Các biến cần có trong .env:
        CLOUDINARY_CLOUD_NAME
        CLOUDINARY_API_KEY
        CLOUDINARY_API_SECRET
    Trả về True nếu thành công.
    """
    global _cloudinary_ok

    missing = [
        name for name, val in {
            "CLOUDINARY_CLOUD_NAME": CLOUDINARY_CLOUD_NAME,
            "CLOUDINARY_API_KEY":    CLOUDINARY_API_KEY,
            "CLOUDINARY_API_SECRET": CLOUDINARY_API_SECRET,
        }.items() if not val
    ]
    if missing:
        print(f"[WARN] Cloudinary: thiếu biến môi trường: {', '.join(missing)}")
        print("[WARN] Cloudinary bị tắt.")
        return False

    try:
        import cloudinary
        import cloudinary.uploader

        cloudinary.config(
            cloud_name = CLOUDINARY_CLOUD_NAME,
            api_key    = CLOUDINARY_API_KEY,
            api_secret = CLOUDINARY_API_SECRET,
            secure     = True,
        )
        os.makedirs(LOCAL_SNAPSHOT_DIR, exist_ok=True)
        _cloudinary_ok = True
        print(f"[CLOUDINARY] Khởi tạo thành công (cloud={CLOUDINARY_CLOUD_NAME}).")
        return True

    except Exception as e:
        print(f"[CLOUDINARY ERROR] init: {e}")
        return False


# ============================================================
# CHỤP ẢNH
# ============================================================

def capture_snapshot(cap: cv2.VideoCapture, bin_id: str, bin_type: str) -> Optional[str]:
    """
    Chụp 1 frame từ webcam và lưu vào LOCAL_SNAPSHOT_DIR.

    Args:
        cap:      cv2.VideoCapture đang mở.
        bin_id:   ID thùng rác (vd: "bin_001") — dùng trong tên file.
        bin_type: Ngăn rác (ORGANIC / RECYCLABLE / HAZARDOUS / OTHER).

    Returns:
        Đường dẫn file ảnh local nếu thành công, None nếu thất bại.
    """
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[CLOUDINARY] Không chụp được ảnh từ webcam.")
        return None

    os.makedirs(LOCAL_SNAPSHOT_DIR, exist_ok=True)
    timestamp  = int(time.time() * 1000)
    filename   = f"{bin_id}_{bin_type}_{timestamp}.jpg"
    local_path = os.path.join(LOCAL_SNAPSHOT_DIR, filename)

    cv2.imwrite(local_path, frame)
    print(f"[CLOUDINARY] Ảnh đã lưu local: {local_path}")
    return local_path


# ============================================================
# UPLOAD ẢNH + GHI LOG FIRESTORE
# ============================================================

def upload_and_log(
    cloudinary_ok:    bool,
    firebase_ok:      bool,
    firestore_client,               # firebase_admin.firestore.client()
    bin_id:           str,
    bin_type:         str,
    locked_class:     str,
    confidence_score: float,
    local_image_path: Optional[str],
    on_done_callback: Optional[callable] = None,
):
    """
    Upload ảnh lên Cloudinary rồi ghi classification_logs vào Firestore.
    Chạy trong thread riêng.

    Args:
        cloudinary_ok:    Kết quả từ init_cloudinary().
        firebase_ok:      Kết quả từ init_firebase() trong firebaseUtil.
        firestore_client: firestore.client() từ firebase_admin.
        bin_id:           ID thùng rác.
        bin_type:         Ngăn rác (ORGANIC / RECYCLABLE / HAZARDOUS / OTHER).
        locked_class:     Class rác (vd: "Plastic", "Battery", ...).
        confidence_score: Độ tin cậy cuối cùng của model (0.0 – 1.0).
        local_image_path: Đường dẫn ảnh local để upload (từ capture_snapshot).
                          None → log không có image_url.
        on_done_callback: Gọi sau khi upload + log xong.
    """

    def _run():
        image_url = None
        log_id    = f"cls_{uuid.uuid4().hex[:8]}"

        # ── 1. Upload Cloudinary ──────────────────────────────────────
        if cloudinary_ok and local_image_path and os.path.exists(local_image_path):
            try:
                import cloudinary.uploader

                folder = f"{SAVE_ROOT_DIR}/{BIN_ID}/{BIN_TO_CLOUDINARY_FOLDER.get(bin_type, 'General')}"
                pub_id = f"{bin_id}_{int(time.time() * 1000)}"

                result = cloudinary.uploader.upload(
                    local_image_path,
                    folder        = folder,   # "PBL5/{bin_id}/{category}"
                    public_id     = pub_id,
                    overwrite     = True,
                    resource_type = "image",
                )
                image_url = result.get("secure_url")
                print(f"[CLOUDINARY] Upload thành công: {image_url}")

                # Xoá ảnh local sau khi upload xong
                os.remove(local_image_path)
                print(f"[CLOUDINARY] Đã xoá ảnh local: {local_image_path}")

            except Exception as e:
                print(f"[CLOUDINARY ERROR] upload: {e}")
        else:
            if local_image_path:
                print("[CLOUDINARY] Skip upload (cloudinary offline hoặc file không tồn tại).")
            # Nếu không upload được vẫn cần xoá file để tránh tích lũy
            if local_image_path and os.path.exists(local_image_path):
                try:
                    os.remove(local_image_path)
                except OSError:
                    pass

        # ── 2. Ghi Firestore: classification_logs ─────────────────────
        if firebase_ok and firestore_client is not None:
            try:
                from firebase_admin import firestore as fs_module

                doc_data = {
                    "log_id":               log_id,
                    "bin_id":               bin_id,
                    "classification_result": locked_class,
                    "confidence_score":     round(confidence_score, 4),
                    "classified_at":        fs_module.SERVER_TIMESTAMP,
                    "image_url":            image_url or "",
                }

                firestore_client.collection("classification_logs").document(log_id).set(doc_data)
                print(f"[FIRESTORE] Ghi classification_logs/{log_id} thành công.")

            except Exception as e:
                print(f"[FIREBASE ERROR][FIRESTORE] classification_logs: {e}")
        else:
            print("[FIRESTORE] Skip log (firebase offline).")

        if on_done_callback:
            on_done_callback()

    threading.Thread(target=_run, daemon=True).start()


# ============================================================
# HELPER: lấy firestore_client từ firebase_admin (tiện dùng ở testPC.py)
# ============================================================

def get_firestore_client():
    """
    Trả về firestore.client() nếu Firebase đã được init, ngược lại trả None.
    Gọi SAU khi firebaseUtil.init_firebase() đã chạy thành công.
    """
    try:
        from firebase_admin import firestore
        return firestore.client()
    except Exception as e:
        print(f"[FIRESTORE] get_client error: {e}")
        return None