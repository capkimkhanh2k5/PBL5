#!/bin/bash
# install_pi4.sh - Cài đặt SmartBin BLE Server trên Raspberry Pi 4
set -e

echo "========================================="
echo "  SmartBin BLE Server - Pi4 Installer"
echo "========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_VENV="${SMARTBIN_VENV:-$SCRIPT_DIR/../tflite-env}"

if [ "$EUID" -ne 0 ]; then
    echo "❌ Chạy script này với sudo"
    exit 1
fi

# Cập nhật package list
echo "[1/5] Cập nhật apt..."
apt-get update -q

# Cài đặt dependencies
echo "[2/5] Cài đặt dependencies..."
apt-get install -y python3-pip python3-venv python3-dbus bluez bluetooth network-manager libdbus-1-dev libglib2.0-dev

# Bật các service cần thiết cho BLE và WiFi provisioning
systemctl enable bluetooth
systemctl start bluetooth
systemctl enable NetworkManager
systemctl start NetworkManager
if command -v rfkill >/dev/null 2>&1; then
    rfkill unblock bluetooth || true
    rfkill unblock wifi || true
fi

mkdir -p /opt/smartbin /etc/smartbin

# Cài đặt Python packages
echo "[3/5] Cài đặt Python packages..."
if [ -x "$PROJECT_VENV/bin/python" ]; then
    VENV_PATH="$(realpath "$PROJECT_VENV")"
    echo "Dùng venv có sẵn: $VENV_PATH"
    if [ -e /opt/smartbin/venv ] && [ ! -L /opt/smartbin/venv ]; then
        mv /opt/smartbin/venv "/opt/smartbin/venv.backup.$(date +%Y%m%d%H%M%S)"
    fi
    ln -sfn "$VENV_PATH" /opt/smartbin/venv
else
    echo "Không tìm thấy $PROJECT_VENV, tạo venv mới tại /opt/smartbin/venv"
    python3 -m venv --system-site-packages /opt/smartbin/venv
fi
/opt/smartbin/venv/bin/python -m pip install --upgrade pip
/opt/smartbin/venv/bin/python -m pip install bluezero dbus-python

# Tạo thư mục và copy files
echo "[4/5] Copy files..."
cp "$SCRIPT_DIR/smartbin_ble_server.py" /opt/smartbin/
chmod +x /opt/smartbin/smartbin_ble_server.py

# Tạo config mặc định nếu chưa có
if [ ! -f /etc/smartbin/config.json ]; then
    DEVICE_ID="SMARTBIN-$(hostname | tr '[:lower:]' '[:upper:]')"
    SECRET_KEY="SB-$(openssl rand -hex 8 | tr '[:lower:]' '[:upper:]')"
    cat > /etc/smartbin/config.json << EOF
{
    "device_id": "$DEVICE_ID",
    "secret_key": "$SECRET_KEY",
    "ble_device_name": "$DEVICE_ID",
    "max_auth_attempts": 5,
    "lockout_seconds": 60,
    "main_service": "smartbin-main.service",
    "start_main_after_wifi": false,
    "stop_ble_after_wifi": false
}
EOF
    chmod 600 /etc/smartbin/config.json
    echo ""
    echo "⚠️  CONFIG TẠO TỰ ĐỘNG:"
    echo "   Device ID  : $DEVICE_ID"
    echo "   Secret Key : $SECRET_KEY"
    echo "   ➡️  Lưu Secret Key này để cấu hình App!"
    echo ""
fi

# Cài đặt systemd service
echo "[5/5] Cài đặt systemd service..."
cp "$SCRIPT_DIR/smartbin-ble.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable smartbin-ble
systemctl start smartbin-ble

echo ""
echo "✅ Cài đặt hoàn tất!"
echo "   Status: sudo systemctl status smartbin-ble"
echo "   Logs:   sudo journalctl -u smartbin-ble -f"
echo ""
echo "📋 Config hiện tại:"
cat /etc/smartbin/config.json
