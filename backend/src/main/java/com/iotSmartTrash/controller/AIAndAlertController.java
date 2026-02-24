package com.iotSmartTrash.controller;

import com.iotSmartTrash.dto.AlertCreateDTO;
import com.iotSmartTrash.dto.AlertResolveDTO;
import com.iotSmartTrash.dto.AlertResponseDTO;
import com.iotSmartTrash.dto.ClassificationLogCreateDTO;
import com.iotSmartTrash.model.Alert;
import com.iotSmartTrash.model.ClassificationLog;
import com.iotSmartTrash.service.AlertService;
import com.iotSmartTrash.service.ClassificationLogService;
import com.iotSmartTrash.service.FirebaseStorageService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

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
    private final FirebaseStorageService storageService;
    private final ObjectMapper objectMapper;

    /** Nhận log AI và hình ảnh từ Raspberry Pi */
    @PostMapping(value = "/classification-logs", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<String> saveAILog(
            @RequestPart("file") MultipartFile file,
            @RequestPart("log") String logJson) {
        try {
            ClassificationLogCreateDTO dto = objectMapper.readValue(logJson, ClassificationLogCreateDTO.class);
            ClassificationLog log = dto.toModel();
            log.setImageUrl(storageService.uploadImage(file));
            String updateTime = aiLogger.saveLog(log);
            return ResponseEntity.ok("AI Data received and saved at " + updateTime);
        } catch (Exception e) {
            return ResponseEntity.status(500).body("Error processing classification log: " + e.getMessage());
        }
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
