Ứng dụng di động bao gồm các trang chính sau:

- Trang đăng nhập (Login Screen) – Cho phép người dùng đăng nhập vào hệ thống.

- Trang đăng ký (Register Screen) – Tạo tài khoản người dùng mới.

- Trang chính (Home Screen) – Hiển thị tổng quan các thùng rác thông minh.

- Trang chi tiết thùng rác (Bin Detail Screen) – Xem thông tin và mức độ đầy của thùng rác.

- Trang lịch sử (History Screen) – Hiển thị lịch sử đổ rác hoặc hoạt động của người dùng.

- Trang lịch thu gom (Schedule Screen) – Xem lịch thu gom rác.

- Trang quét QR (Scan QR Screen) – Quét mã QR trên thùng rác để tương tác nhanh.

- Trang AI Chat (AI Chat Screen) – Hỗ trợ người dùng hỏi về phân loại rác hoặc thông tin liên quan.

Ngoài ra, ứng dụng còn có:

- Main Shell – Quản lý điều hướng giữa các trang chính trong ứng dụng.

- Theme / App Background – Quản lý giao diện và màu sắc chung của ứng dụng.

Database Description

1. bins_metadata

Thông tin cố định của thùng rác

- name: tên thùng (VD: SMART_BIN_01)
- location_description: mô tả vị trí (VD: Sảnh 1)
- latitude: vĩ độ (GPS)
- longitude: kinh độ (GPS)
- installed_at: thời gian lắp đặt (timestamp)

 Document ID = binId (ID chính của thùng)

 2. bin_raw_sensor_logs

Dữ liệu sensor thô từ Raspberry Pi (real-time)

- fill_organic: mức đầy rác hữu cơ (%)
- fill_recycle: mức đầy rác tái chế (%)
- fill_non_recycle: mức đầy rác không tái chế (%)
- fill_hazardous: mức đầy rác nguy hại (%)
- recorded_at: thời gian gửi dữ liệu (timestamp)

 Cấu trúc:

bin_raw_sensor_logs/{binId}/logs/{logId}
 3. bin_sensor_logs

Dữ liệu đã xử lý (trung bình, dùng cho thống kê & ML)

- bin_id: ID thùng (liên kết bins_metadata)
- avg_fill_organic: trung bình rác hữu cơ
- avg_fill_recycle: trung bình rác tái chế
- avg_fill_non_recycle: trung bình rác không tái chế
- avg_fill_hazardous: trung bình rác nguy hại
- avg_battery: pin trung bình thiết bị (%)
- sample_count: số mẫu dùng để tính
- date: ngày (VD: 2026-03-23)
- period: khoảng thời gian (VD: H12)
- recorded_at: thời gian ghi log
 4. bin_realtime_status

Trạng thái hiện tại của thùng (real-time)

- temperature: nhiệt độ trong thùng (°C)
- fillOrganic: mức đầy rác hữu cơ (%)
- fillRecycle: mức đầy rác tái chế (%)
- fillNonRecycle: mức đầy rác không tái chế (%)
- fillHazardous: mức đầy rác nguy hại (%)
- status: trạng thái (OK / FULL / UNKNOWN)
- lastUpdated: thời gian cập nhật gần nhất

 Document ID = binId

 5. classification_logs

Kết quả AI phân loại rác

- bin_id: ID thùng
- classification_result: kết quả phân loại (Organic, Recycle, …)
- confidence_score: độ tin cậy của model
- image_url: link ảnh đầu vào
- classified_at: thời gian phân loại
- log_id: ID log
 6. alerts

Cảnh báo khi thùng đầy hoặc có sự cố

- alert_type: loại cảnh báo (FULL_BIN, ERROR, …)
- bin_id: ID thùng
- message: nội dung cảnh báo
- fill_levels_at_alert: mức rác tại thời điểm cảnh báo (map)
- fill_levels_at_resolve: mức rác khi xử lý xong (map / null)
- created_at: thời gian tạo cảnh báo
- resolved_at: thời gian xử lý (null nếu chưa xử lý)
 7. users

Thông tin người dùng hệ thống

- username: tên đăng nhập
- email: email
- avatar_url: ảnh đại diện
- created_at: thời gian tạo tài khoản

 Document ID = userId
