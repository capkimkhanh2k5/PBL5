package com.iotSmartTrash.controller;

import com.iotSmartTrash.model.Alert;
import com.iotSmartTrash.model.ClassificationLog;
import com.iotSmartTrash.service.AlertService;
import com.iotSmartTrash.service.ClassificationLogService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

@RestController
@RequestMapping("/api/v1/system")
@RequiredArgsConstructor
public class AIAndAlertController {

    private final ClassificationLogService aiLogger;
    private final AlertService alertService;

    // Nhận log từ Raspberry Pi Camera Model chuyển về
    @PostMapping("/classification-logs")
    public ResponseEntity<String> saveAILog(@RequestBody ClassificationLog log)
            throws ExecutionException, InterruptedException {
        String updateTime = aiLogger.saveLog(log);
        return ResponseEntity.ok("AI Data received and saved at " + updateTime);
    }

    // Lấy toàn bộ cảnh báo trên Admin Portal
    @GetMapping("/alerts")
    public ResponseEntity<List<Alert>> getAllAlerts() throws ExecutionException, InterruptedException {
        return ResponseEntity.ok(alertService.getAllAlerts());
    }

    // RaspBerry Pi tự động ném Alert (Nhiệt cao quá) lên Server
    @PostMapping("/alerts")
    public ResponseEntity<String> triggerAlert(@RequestBody Alert alert)
            throws ExecutionException, InterruptedException {
        String updateTime = alertService.createAlert(alert);
        return ResponseEntity.ok("Alert generated at " + updateTime);
    }

    // Staff/Admin bấm "Giải quyết cảnh báo"
    @PatchMapping("/alerts/{id}/resolve")
    public ResponseEntity<String> resolveAlert(@PathVariable String id, @RequestBody Map<String, String> body)
            throws ExecutionException, InterruptedException {
        String resolvedBy = body.get("resolved_by"); // Lấy UID người xử lý
        String updateTime = alertService.resolveAlert(id, resolvedBy);
        return ResponseEntity.ok("Alert Resolved Successfully by " + resolvedBy + " at " + updateTime);
    }
}
