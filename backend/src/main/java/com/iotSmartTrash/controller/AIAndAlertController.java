package com.iotSmartTrash.controller;

import com.iotSmartTrash.dto.AlertCreateDTO;
import com.iotSmartTrash.dto.AlertResolveDTO;
import com.iotSmartTrash.dto.AlertResponseDTO;
import com.iotSmartTrash.dto.BinPickupScheduleDTO;
import com.iotSmartTrash.dto.ClassificationLogCreateDTO;
import com.iotSmartTrash.model.Alert;
import com.iotSmartTrash.model.ClassificationLog;
import com.iotSmartTrash.service.AlertService;
import com.iotSmartTrash.service.BinScheduleService;
import com.iotSmartTrash.service.ClassificationLogService;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import lombok.RequiredArgsConstructor;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v1/system")
@RequiredArgsConstructor
@Validated
public class AIAndAlertController {

    private final ClassificationLogService aiLogger;
    private final AlertService alertService;
    private final BinScheduleService binScheduleService;

    /** Nhận binId, URL ảnh và kết quả phân loại từ AI */
    @PostMapping(value = "/classification-logs", consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<String> saveAILog(
            @Valid @RequestBody ClassificationLogCreateDTO dto) {
        ClassificationLog log = dto.toModel();
        String updateTime = aiLogger.saveLog(log);
        return ResponseEntity.ok("AI data received and saved at " + updateTime);
    }

    /** Lấy toàn bộ cảnh báo cho Admin Portal */
    @GetMapping("/alerts")
    public ResponseEntity<List<AlertResponseDTO>> getAllAlerts() {
        List<Alert> alerts = alertService.getAllAlerts();
        List<AlertResponseDTO> dtos = alerts.stream()
                .map(AlertResponseDTO::fromModel)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }

    /** Lấy lịch sử phân loại (mới nhất trước), có thể filter theo binId */
    @GetMapping("/classification-logs")
    public ResponseEntity<List<ClassificationLog>> getClassificationLogs(
            @RequestParam(required = false) String binId,
            @RequestParam(defaultValue = "20") int limit) {
        int safeLimit = Math.max(1, Math.min(limit, 100));
        return ResponseEntity.ok(aiLogger.getLatestLogs(binId, safeLimit));
    }

    /** Lấy lịch thu gom dựa trên dữ liệu aggregate bin_sensor_logs */
    @GetMapping("/pickup-schedule")
    public ResponseEntity<List<BinPickupScheduleDTO>> getPickupSchedule(
            @RequestParam(defaultValue = "40") int limit) {
        int safeLimit = Math.max(1, Math.min(limit, 100));
        return ResponseEntity.ok(binScheduleService.getPickupSchedule(safeLimit));
    }

    /** Nhận tín hiệu cảnh báo từ IoT Sensor */
    @PostMapping("/alerts")
    public ResponseEntity<String> triggerAlert(@Valid @RequestBody AlertCreateDTO alertDto) {
        Alert alert = alertDto.toModel();
        String updateTime = alertService.createAlert(alert);
        return ResponseEntity.ok("Alert received and processed at " + updateTime);
    }

    /** Đánh dấu cảnh báo đã giải quyết */
    @PatchMapping("/alerts/{id}/resolve")
    public ResponseEntity<String> resolveAlert(
            @PathVariable String id,
            @Valid @RequestBody AlertResolveDTO body) {
        String updateTime = alertService.resolveAlert(id, body.getResolvedBy());
        return ResponseEntity.ok("Successfully marked alert as resolved at " + updateTime);
    }
}
