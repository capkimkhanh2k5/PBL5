#!/usr/bin/env python3
"""
SmartBin BLE WiFi Setup Server - Raspberry Pi 4
================================================
Chạy trên Raspberry Pi 4, hoạt động như BLE GATT Server.
Cho phép thiết bị client gửi WiFi credentials sau khi xác thực secret key.

Cài đặt:
    pip install dbus-python bluezero

Chạy:
    sudo python3 smartbin_ble_server.py
"""

import sys
import os
import json
import logging
import hashlib
import hmac
import shutil
import subprocess
import threading
import time
from pathlib import Path

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/var/log/smartbin_ble.log"),
    ],
)
log = logging.getLogger("SmartBinBLE")

# ─── Config ─────────────────────────────────────────────────────────────────
CONFIG_FILE = Path("/etc/smartbin/config.json")
DEFAULT_CONFIG = {
    "device_id": "SMARTBIN-001",
    "secret_key": "SB-CHANGEME-2024",   # ← Thay bằng key thực tế của từng thiết bị
    "ble_device_name": "SmartBin-001",
    "max_auth_attempts": 5,
    "lockout_seconds": 60,
    "main_service": "smartbin-main.service",
    "start_main_after_wifi": False,
    "stop_ble_after_wifi": False,
}

# ─── BLE UUIDs ───────────────────────────────────────────────────────────────
# Service UUID cho SmartBin WiFi Setup
WIFI_SETUP_SERVICE_UUID      = "12345678-1234-5678-1234-56789abcdef0"
# Characteristic: Client gửi secret key lên đây
AUTH_KEY_CHAR_UUID           = "12345678-1234-5678-1234-56789abcdef1"
# Characteristic: Client gửi WiFi SSID
WIFI_SSID_CHAR_UUID          = "12345678-1234-5678-1234-56789abcdef2"
# Characteristic: Client gửi WiFi Password
WIFI_PASS_CHAR_UUID          = "12345678-1234-5678-1234-56789abcdef3"
# Characteristic: Client đọc trạng thái + kết quả (notify)
STATUS_CHAR_UUID             = "12345678-1234-5678-1234-56789abcdef4"
# Characteristic: Client đọc thông tin thiết bị
DEVICE_INFO_CHAR_UUID        = "12345678-1234-5678-1234-56789abcdef5"

# ─── Status Codes ────────────────────────────────────────────────────────────
STATUS = {
    "IDLE":            "00",
    "AUTH_OK":         "01",
    "AUTH_FAIL":       "02",
    "AUTH_LOCKED":     "03",
    "WIFI_SAVING":     "04",
    "WIFI_SAVED":      "05",
    "WIFI_FAIL":       "06",
    "WIFI_CONNECTING": "07",
    "WIFI_CONNECTED":  "08",
}

# ─── Auth State ──────────────────────────────────────────────────────────────
class AuthState:
    def __init__(self):
        self.authenticated = False
        self.attempts      = 0
        self.locked_until  = 0
        self._lock         = threading.Lock()

    def is_locked(self):
        with self._lock:
            return time.time() < self.locked_until

    def reset(self):
        with self._lock:
            self.authenticated = False
            self.attempts      = 0

    def authenticate(self, provided_key: str, expected_key: str, max_attempts: int, lockout_sec: int) -> bool:
        with self._lock:
            if time.time() < self.locked_until:
                log.warning("Auth blocked: device locked out")
                return False

            # So sánh hash để tránh timing attack
            h_provided = hashlib.sha256(provided_key.encode()).hexdigest()
            h_expected = hashlib.sha256(expected_key.encode()).hexdigest()

            if hmac.compare_digest(h_provided, h_expected):
                self.authenticated = True
                self.attempts      = 0
                log.info("✅ Authentication successful")
                return True
            else:
                self.attempts += 1
                log.warning(f"❌ Auth failed ({self.attempts}/{max_attempts})")
                if self.attempts >= max_attempts:
                    self.locked_until = time.time() + lockout_sec
                    self.authenticated = False
                    log.warning(f"🔒 Device locked for {lockout_sec}s")
                return False


# ─── WiFi Manager ────────────────────────────────────────────────────────────
class WiFiManager:
    @staticmethod
    def _run(cmd, timeout=20):
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    @staticmethod
    def _nmcli_available() -> bool:
        return shutil.which("nmcli") is not None

    @staticmethod
    def _verify_connected(ssid: str) -> bool:
        result = WiFiManager._run(
            ["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"],
            timeout=10,
        )
        if result.returncode != 0:
            log.warning(f"nmcli verify warning: {result.stderr.strip()}")
            return False
        return any(line == f"yes:{ssid}" for line in result.stdout.splitlines())

    @staticmethod
    def save_and_connect(ssid: str, password: str) -> tuple[bool, str]:
        """Lưu WiFi profile và kết nối bằng nmcli (NetworkManager)."""
        log.info(f"Saving WiFi: SSID='{ssid}'")
        if not WiFiManager._nmcli_available():
            return False, "nmcli not found. Install and enable NetworkManager."

        try:
            WiFiManager._run(["nmcli", "radio", "wifi", "on"], timeout=10)
            WiFiManager._run(["nmcli", "device", "wifi", "rescan"], timeout=20)

            # Xóa profile cũ nếu có
            WiFiManager._run(
                ["nmcli", "connection", "delete", ssid],
                timeout=10,
            )
            # Tạo profile mới
            cmd = ["nmcli", "device", "wifi", "connect", ssid]
            if password:
                cmd.extend(["password", password])
            result = WiFiManager._run(cmd, timeout=45)

            if result.returncode == 0 and WiFiManager._verify_connected(ssid):
                log.info(f"✅ WiFi connected: {ssid}")
                return True, ""

            err = (result.stderr or result.stdout or "WiFi connection failed").strip()
            log.error(f"nmcli error: {err}")
            return False, err
        except subprocess.TimeoutExpired:
            log.error("WiFi connection timeout")
            return False, "WiFi connection timeout"
        except Exception as e:
            log.error(f"WiFi error: {e}")
            return False, str(e)

    @staticmethod
    def save_credentials_to_file(ssid: str, password: str):
        """Lưu credentials vào file cấu hình để dùng khi khởi động lại."""
        wifi_config_path = Path("/etc/smartbin/wifi_credentials.json")
        wifi_config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"ssid": ssid, "password": password, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(wifi_config_path, "w") as f:
            json.dump(data, f, indent=2)
        os.chmod(wifi_config_path, 0o600)
        log.info(f"WiFi credentials saved to {wifi_config_path}")


# ─── Config Loader ───────────────────────────────────────────────────────────
def load_config() -> dict:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        log.info(f"Config loaded: device_id={cfg.get('device_id')}")
        return {**DEFAULT_CONFIG, **cfg}
    else:
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        os.chmod(CONFIG_FILE, 0o600)
        log.warning(f"Default config created at {CONFIG_FILE}. Please update secret_key!")
        return DEFAULT_CONFIG.copy()


# ─── BLE Server (bluezero) ───────────────────────────────────────────────────
try:
    from bluezero import adapter, peripheral, async_tools
    BLUEZERO_AVAILABLE = True
except ImportError:
    BLUEZERO_AVAILABLE = False
    log.warning("bluezero not installed. Running in DEMO mode (no real BLE).")


class SmartBinBLEServer:
    def __init__(self):
        self.config     = load_config()
        self.auth       = AuthState()
        self.wifi_mgr   = WiFiManager()
        self._ssid_buf  = ""
        self._pass_buf  = ""
        self._status    = STATUS["IDLE"]
        self._last_error = ""
        self._wifi_lock = threading.Lock()
        self._app       = None

    # ── BLE Characteristic Callbacks ─────────────────────────────────────────

    def on_auth_key_write(self, value, options):
        """Callback khi client gửi secret key."""
        key = bytes(value).decode("utf-8", errors="ignore").strip()
        log.info("Received auth key attempt")
        self._last_error = ""

        if self.auth.is_locked():
            self._status = STATUS["AUTH_LOCKED"]
            self._last_error = "Too many failed attempts. Try again later."
            log.warning("Auth attempt while locked")
            return

        ok = self.auth.authenticate(
            key,
            self.config["secret_key"],
            self.config["max_auth_attempts"],
            self.config["lockout_seconds"],
        )
        self._status = STATUS["AUTH_OK"] if ok else STATUS["AUTH_FAIL"]
        # Reset buffers mỗi lần auth
        self._ssid_buf = ""
        self._pass_buf = ""

    def on_ssid_write(self, value, options):
        """Callback khi client gửi SSID."""
        if not self.auth.authenticated:
            log.warning("Unauthorized SSID write attempt")
            self._status = STATUS["AUTH_FAIL"]
            self._last_error = "Not authenticated"
            return
        self._ssid_buf = bytes(value).decode("utf-8", errors="ignore").strip()
        if not self._ssid_buf or len(self._ssid_buf.encode("utf-8")) > 32:
            log.warning("Invalid SSID received")
            self._ssid_buf = ""
            self._status = STATUS["WIFI_FAIL"]
            self._last_error = "SSID must be 1-32 bytes"
            return
        log.info(f"SSID received: '{self._ssid_buf}'")
        self._last_error = ""

    def on_password_write(self, value, options):
        """Callback khi client gửi Password - trigger kết nối WiFi."""
        if not self.auth.authenticated:
            log.warning("Unauthorized password write attempt")
            self._status = STATUS["AUTH_FAIL"]
            self._last_error = "Not authenticated"
            return

        self._pass_buf = bytes(value).decode("utf-8", errors="ignore")
        log.info("Password received, starting WiFi setup...")
        self._status = STATUS["WIFI_SAVING"]
        self._last_error = ""

        # Chạy WiFi setup trong thread riêng để không block BLE
        threading.Thread(target=self._do_wifi_setup, daemon=True).start()

    def _do_wifi_setup(self):
        with self._wifi_lock:
            if not self._ssid_buf:
                log.error("No SSID set before password write")
                self._status = STATUS["WIFI_FAIL"]
                self._last_error = "No SSID set before password"
                return

            self._status = STATUS["WIFI_CONNECTING"]

            # NetworkManager sẽ lưu profile nếu kết nối thành công.
            success, error = self.wifi_mgr.save_and_connect(self._ssid_buf, self._pass_buf)
            if success:
                self.wifi_mgr.save_credentials_to_file(self._ssid_buf, self._pass_buf)
                self._status = STATUS["WIFI_CONNECTED"]
                self._last_error = ""
                self._start_main_service()
            else:
                self._status = STATUS["WIFI_FAIL"]
                self._last_error = error

            # Reset auth session sau khi setup xong
            self.auth.reset()

    def _start_main_service(self):
        if not self.config.get("start_main_after_wifi", False):
            return

        service = self.config.get("main_service", "smartbin-main.service")
        try:
            result = subprocess.run(
                ["systemctl", "start", service],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                log.info(f"Main service started: {service}")
            else:
                log.error(f"Failed to start {service}: {result.stderr.strip()}")

            if self.config.get("stop_ble_after_wifi", False):
                subprocess.Popen(["systemctl", "stop", "smartbin-ble.service"])
        except Exception as e:
            log.error(f"Main service trigger error: {e}")

    def on_status_read(self) -> bytes:
        """Callback khi client đọc status."""
        payload = json.dumps({
            "code":   self._status,
            "device": self.config["device_id"],
            "error":  self._last_error,
            "ts":     int(time.time()),
        })
        return list(payload.encode("utf-8"))

    def on_device_info_read(self) -> bytes:
        """Callback khi client đọc thông tin thiết bị."""
        info = json.dumps({
            "device_id":   self.config["device_id"],
            "firmware":    "1.0.0",
            "type":        "SmartBin",
            "ble_version": "5.0",
        })
        return list(info.encode("utf-8"))

    # ── Start Server ─────────────────────────────────────────────────────────

    def start(self):
        if not BLUEZERO_AVAILABLE:
            self._demo_mode()
            return

        log.info(f"Starting BLE server: '{self.config['ble_device_name']}'")
        dongle = adapter.Adapter()
        dongle.powered = True

        self._app = peripheral.Peripheral(
            dongle.address,
            local_name=self.config["ble_device_name"],
        )

        # Thêm service
        self._app.add_service(srv_id=1, uuid=WIFI_SETUP_SERVICE_UUID, primary=True)

        # Auth Key characteristic (WRITE)
        self._app.add_characteristic(
            srv_id=1, chr_id=1, uuid=AUTH_KEY_CHAR_UUID,
            value=[], notifying=False,
            flags=["write", "write-without-response"],
            write_callback=self.on_auth_key_write,
        )

        # SSID characteristic (WRITE)
        self._app.add_characteristic(
            srv_id=1, chr_id=2, uuid=WIFI_SSID_CHAR_UUID,
            value=[], notifying=False,
            flags=["write", "write-without-response"],
            write_callback=self.on_ssid_write,
        )

        # Password characteristic (WRITE)
        self._app.add_characteristic(
            srv_id=1, chr_id=3, uuid=WIFI_PASS_CHAR_UUID,
            value=[], notifying=False,
            flags=["write", "write-without-response"],
            write_callback=self.on_password_write,
        )

        # Status characteristic (READ + NOTIFY)
        self._app.add_characteristic(
            srv_id=1, chr_id=4, uuid=STATUS_CHAR_UUID,
            value=list(STATUS["IDLE"].encode()), notifying=True,
            flags=["read", "notify"],
            read_callback=self.on_status_read,
        )

        # Device Info characteristic (READ)
        self._app.add_characteristic(
            srv_id=1, chr_id=5, uuid=DEVICE_INFO_CHAR_UUID,
            value=[], notifying=False,
            flags=["read"],
            read_callback=self.on_device_info_read,
        )

        self._app.publish()
        log.info("✅ BLE server running. Waiting for connections...")
        async_tools.run_forever()

    def _demo_mode(self):
        """Demo mode khi không có bluezero - simulate luồng xác thực."""
        log.info("=== DEMO MODE (No BLE hardware) ===")
        log.info(f"Device ID: {self.config['device_id']}")
        log.info(f"Secret Key: {self.config['secret_key']}")

        print("\n--- SmartBin BLE Demo Simulator ---")
        key = input("Enter secret key: ").strip()

        ok = self.auth.authenticate(
            key,
            self.config["secret_key"],
            self.config["max_auth_attempts"],
            self.config["lockout_seconds"],
        )
        self._status = STATUS["AUTH_OK"] if ok else STATUS["AUTH_FAIL"]

        if not ok:
            print(f"❌ Authentication failed! Status: {self._status}")
            return

        print(f"✅ Auth OK! Status: {self._status}")
        ssid = input("Enter WiFi SSID: ").strip()
        self._ssid_buf = ssid
        pw   = input("Enter WiFi Password: ").strip()
        self._pass_buf = pw

        self._status = STATUS["WIFI_SAVING"]
        print(f"Saving WiFi: SSID='{ssid}'")
        self.wifi_mgr.save_credentials_to_file(ssid, pw)
        print(f"✅ Credentials saved! Status: {self._status}")


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.geteuid() != 0 and BLUEZERO_AVAILABLE:
        log.error("BLE server requires root. Run with sudo.")
        sys.exit(1)

    server = SmartBinBLEServer()
    try:
        server.start()
    except KeyboardInterrupt:
        log.info("BLE server stopped.")
