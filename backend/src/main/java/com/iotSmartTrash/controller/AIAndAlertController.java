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
import org.springframework.web.multipart.MultipartFile;
import org.springframework.http.MediaType;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.iotSmartTrash.service.FirebaseStorageService;

@RestController
@RequestMapping("/api/v1/system")
@RequiredArgsConstructor
public class AIAndAlertController {

    private final ClassificationLogService aiLogger;
    private final AlertService alertService;
    private final FirebaseStorageService storageService;
    private final ObjectMapper objectMapper;

    // Nhận log và hình ảnh từ Raspberry Pi Camera Model chuyển về
    @PostMapping(value = "/classification-logs", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<String> saveAILog(
            @RequestPart("file") MultipartFile file,
            @RequestPart("log") String logJson) {
        try {
            ClassificationLog log = objectMapper.readValue(logJson, ClassificationLog.class);
            String imageUrl = storageService.uploadImage(file);
            log.setImage_url(imageUrl);

            String updateTime = aiLogger.saveLog(log);
            return ResponseEntity.ok("AI Data received and saved at " + updateTime);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("Error processing classification log: " + e.getMessage());
        }
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
