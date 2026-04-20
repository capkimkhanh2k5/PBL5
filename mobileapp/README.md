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

1. users
 Lưu thông tin người dùng

- email: email đăng nhập
- username: tên hiển thị
- role: quyền (USER / ADMIN)
- avatar_url: ảnh đại diện
- created_at: thời gian tạo

2. bins_metadata
  Lưu thông tin thùng rác: Hiển thị bản đồ, thời gian

- id: mã thùng (bin_001…)
- name: tên thùng
- location_description: vị trí (VD: Sảnh 1)
- latitude: vĩ độ
- longitude: kinh độ
- installed_at: thời gian lắp đặt

3. bin_realtime_status

 Trạng thái hiện tại của thùng (real-time): mức đầy

- temperature: nhiệt độ
- fillOrganic: mức đầy rác hữu cơ
- fillRecycle: mức đầy rác tái chế
- fillNonRecycle: mức đầy rác không tái chế
- fillHazardous: mức đầy rác nguy hại
- status: trạng thái (OK / FULL / UNKNOWN)
- lastUpdated: thời gian cập nhật


5. classification_logs
 Lịch sử phân loại rác (AI)

- log_id: mã log
- bin_id: mã thùng
- image_url: ảnh đầu vào
- classification_result: kết quả (Organic / Recycle…)
- confidence_score: độ chính xác
- classified_at: thời gian phân loại

6. alerts
 Cảnh báo hệ thống, thông báo

- id: mã cảnh báo
- alert_type: loại cảnh báo (FULL_BIN…)
- bin_id: mã thùng
- created_at: thời gian tạo
- fill_levels_at_alert: mức rác khi cảnh báo
- fill_levels_at_resolve: mức rác khi xử lý (nếu có)
