# Tài liệu API cho Raspberry Pi

## Thông tin kết nối

| Mục | Giá trị |
|---|---|
| **Base URL** | `http://<SERVER_IP>:8080` |
| **Xác thực** | Header `X-IoT-API-Key: <API_KEY>` |
| **Content-Type** | `application/json` |

> ⚠️ **Tất cả request POST đều bắt buộc có header `X-IoT-API-Key`.** Nếu thiếu hoặc sai sẽ trả về `401 Unauthorized`.

---

## API 1 — Gửi dữ liệu Sensor

Raspi gọi **mỗi 1 giờ** để gửi dữ liệu cảm biến.

```
POST /api/v1/iot/bins/{binId}/sensor-logs
```

### Request

**Path Parameters:**

| Param | Mô tả | Ví dụ |
|---|---|---|
| `binId` | ID của thùng rác | `bin_001` |

**Headers:**

```
Content-Type: application/json
X-IoT-API-Key: <API_KEY>
```

**Body:**

```json
{
  "batteryLevel": 85,
  "fillOrganic": 60,
  "fillRecycle": 45,
  "fillNonRecycle": 30,
  "fillHazardous": 10,
  "recordedAt": 1711468800000
}
```

| Field | Type | Bắt buộc | Validation | Mô tả |
|---|---|---|---|---|
| `batteryLevel` | `Integer` | ❌ | `0 – 100` | Phần trăm pin |
| `fillOrganic` | `Integer` | ❌ | `0 – 100` | % đầy ngăn hữu cơ |
| `fillRecycle` | `Integer` | ❌ | `0 – 100` | % đầy ngăn tái chế |
| `fillNonRecycle` | `Integer` | ❌ | `0 – 100` | % đầy ngăn không tái chế |
| `fillHazardous` | `Integer` | ❌ | `0 – 100` | % đầy ngăn nguy hại |
| `recordedAt` | `Long` | ❌ | — | Epoch milliseconds. Nếu `null` → server tự gán thời gian hiện tại |

### Response

```
200 OK
"Sensor log saved at 2024-03-26T12:00:00Z"
```

### Ví dụ cURL

```bash
curl -X POST http://<SERVER_IP>:8080/api/v1/iot/bins/bin_001/sensor-logs \
  -H "Content-Type: application/json" \
  -H "X-IoT-API-Key: <API_KEY>" \
  -d '{
    "batteryLevel": 85,
    "fillOrganic": 60,
    "fillRecycle": 45,
    "fillNonRecycle": 30,
    "fillHazardous": 10
  }'
```

### Ví dụ Python

```python
import requests, time

url = "http://<SERVER_IP>:8080/api/v1/iot/bins/bin_001/sensor-logs"
headers = {
    "Content-Type": "application/json",
    "X-IoT-API-Key": "<API_KEY>"
}
data = {
    "batteryLevel": 85,
    "fillOrganic": 60,
    "fillRecycle": 45,
    "fillNonRecycle": 30,
    "fillHazardous": 10,
    "recordedAt": int(time.time() * 1000)
}

response = requests.post(url, json=data, headers=headers)
print(response.status_code, response.text)
```

---

## API 2 — Gửi kết quả phân loại rác (AI)

Raspi chụp ảnh → chạy model AI → upload ảnh lên storage → gửi URL ảnh + kết quả về BE.

```
POST /api/v1/system/classification-logs
```

### Request

**Headers:**

```
Content-Type: application/json
X-IoT-API-Key: <API_KEY>
```

**Body:**

```json
{
  "binId": "bin_001",
  "imageUrl": "https://storage.googleapis.com/.../waste_image.jpg",
  "classificationResult": "Organic"
}
```

| Field | Type | Bắt buộc | Mô tả |
|---|---|---|---|
| `binId` | `String` | ✅ | ID thùng rác |
| `imageUrl` | `String` | ✅ | URL ảnh rác đã upload lên storage |
| `classificationResult` | `String` | ✅ | Kết quả phân loại: `Organic`, `Recycle`, `NonRecycle`, `Hazardous` |

### Response

```
200 OK
"AI data received and saved at 2024-03-26T12:00:00Z"
```

### Ví dụ cURL

```bash
curl -X POST http://<SERVER_IP>:8080/api/v1/system/classification-logs \
  -H "Content-Type: application/json" \
  -H "X-IoT-API-Key: <API_KEY>" \
  -d '{
    "binId": "bin_001",
    "imageUrl": "https://storage.googleapis.com/.../waste_image.jpg",
    "classificationResult": "Organic"
  }'
```

### Ví dụ Python

```python
import requests

url = "http://<SERVER_IP>:8080/api/v1/system/classification-logs"
headers = {
    "Content-Type": "application/json",
    "X-IoT-API-Key": "<API_KEY>"
}
data = {
  "binId": "bin_001",
    "imageUrl": "https://storage.googleapis.com/.../waste_image.jpg",
    "classificationResult": "Organic"
}

response = requests.post(url, json=data, headers=headers)
print(response.status_code, response.text)
```

---

## API 3 — Gửi cảnh báo bất thường

Raspi gửi cảnh báo khi phát hiện sự cố: thùng đầy, nhiệt độ cao, pin yếu, khói, cháy.

```
POST /api/v1/system/alerts
```

### Request

**Headers:**

```
Content-Type: application/json
X-IoT-API-Key: <API_KEY>
```

**Body:**

```json
{
  "binId": "bin_001",
  "alertType": "FULL_BIN",
  "severity": "CRITICAL",
  "message": "Ngăn hữu cơ đã đầy 95%"
}
```

| Field | Type | Bắt buộc | Giá trị hợp lệ | Mô tả |
|---|---|---|---|---|
| `binId` | `String` | ✅ | — | ID thùng rác |
| `alertType` | `String` | ✅ | `FULL_BIN`, `LOW_BATTERY`, `SMOKE`, `FIRE` | Loại cảnh báo |
| `severity` | `String` | ✅ | `INFO`, `WARNING`, `CRITICAL` | Mức độ nghiêm trọng |
| `message` | `String` | ❌ | — | Mô tả chi tiết (tuỳ chọn) |

### Khi nào gọi API này?

| alertType | Điều kiện | severity gợi ý |
|---|---|---|
| `FULL_BIN` | Bất kỳ ngăn nào `fill >= 90%` | `WARNING` hoặc `CRITICAL` |
| `LOW_BATTERY` | `batteryLevel <= 15%` | `INFO` hoặc `WARNING` |
| `SMOKE` | Cảm biến phát hiện khói | `CRITICAL` |
| `FIRE` | Cảm biến phát hiện cháy | `CRITICAL` |

### Response

```
200 OK
"Alert received and processed at 2024-03-26T12:00:00Z"
```

### Ví dụ Python

```python
import requests

url = "http://<SERVER_IP>:8080/api/v1/system/alerts"
headers = {
    "Content-Type": "application/json",
    "X-IoT-API-Key": "<API_KEY>"
}
data = {
    "binId": "bin_001",
    "alertType": "FULL_BIN",
    "severity": "CRITICAL",
    "message": "Ngăn hữu cơ đã đầy 95%"
}

response = requests.post(url, json=data, headers=headers)
print(response.status_code, response.text)
```
