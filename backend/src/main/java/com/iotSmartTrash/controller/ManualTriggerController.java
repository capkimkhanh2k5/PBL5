package com.iotSmartTrash.controller;

import com.iotSmartTrash.scheduler.SensorLogScheduler;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/trigger")
@RequiredArgsConstructor
public class ManualTriggerController {

    private final SensorLogScheduler scheduler;

    /**
     * API Test
     * Kích hoạt ép chạy tính toán dữ liệu 6 tiếng ngay lập tức mà không cần đợi đến
     * 0h/6h/12h/18h
     */
    @PostMapping("/aggregate-sensor-logs")
    public ResponseEntity<String> triggerAggregateLogs() {
        scheduler.aggregateSensorLogs();
        return ResponseEntity
                .ok("Đã gửi luồng chạy ngầm tổng hợp log cảm biến. Kiểm tra Terminal Console để xem tiến trình.");
    }
}
