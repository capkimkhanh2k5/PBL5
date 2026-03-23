package com.iotSmartTrash.controller;

import com.iotSmartTrash.dto.BinRealtimeStatusResponseDTO;
import com.iotSmartTrash.dto.RawSensorLogCreateDTO;
import com.iotSmartTrash.model.BinRawSensorLog;
import com.iotSmartTrash.model.BinRealtimeStatus;
import com.iotSmartTrash.service.BinRawSensorLogService;
import com.iotSmartTrash.service.BinRealtimeService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Controller cho IoT device (Raspberry Pi) gửi dữ liệu sensor và nhận trạng
 * thái thùng rác từ BE.
 */
@RestController
@RequestMapping("/api/v1/iot")
@RequiredArgsConstructor
@Validated
public class IoTSensorController {

    private final BinRealtimeService binRealtimeService;
    private final BinRawSensorLogService rawSensorLogService;

    // ━━━━━━━━━━━━━━━━━ Bin Realtime Status ━━━━━━━━━━━━━━━━━

    /**
     * Legacy endpoint intentionally disabled.
     * Device must send only /sensor-logs.
     */
    @PostMapping("/bins/{binId}/status")
    public ResponseEntity<String> rejectLegacyStatusWrite(@PathVariable String binId) {
        return ResponseEntity.status(HttpStatus.GONE)
                .body("Deprecated endpoint. Use POST /api/v1/iot/bins/{binId}/sensor-logs only.");
    }

    /**
     * Lấy trạng thái realtime của 1 thùng rác (cho Flutter app).
     */
    @GetMapping("/bins/{binId}/status")
    public ResponseEntity<BinRealtimeStatusResponseDTO> getBinStatus(@PathVariable String binId) {
        BinRealtimeStatus status = binRealtimeService.getStatus(binId);
        return ResponseEntity.ok(BinRealtimeStatusResponseDTO.fromModel(status));
    }

    /**
     * Lấy trạng thái realtime của tất cả thùng rác (cho Flutter app).
     */
    @GetMapping("/bins/status")
    public ResponseEntity<List<BinRealtimeStatusResponseDTO>> getAllBinStatuses() {
        List<BinRealtimeStatus> statuses = binRealtimeService.getAllStatuses();
        List<BinRealtimeStatusResponseDTO> dtos = statuses.stream()
                .map(BinRealtimeStatusResponseDTO::fromModel)
                .collect(Collectors.toList());
        return ResponseEntity.ok(dtos);
    }

    // ━━━━━━━━━━━━━━━━━ Raw Sensor Logs ━━━━━━━━━━━━━━━━━

    /**
     * Raspi gọi mỗi 30 giây để ghi raw sensor data.
     * Replaces RTDB write to: bin_sensor_logs/{bin_id}/{log_id}
     */
    @PostMapping("/bins/{binId}/sensor-logs")
    public ResponseEntity<String> addSensorLog(
            @PathVariable String binId,
            @Valid @RequestBody RawSensorLogCreateDTO dto) {
        BinRawSensorLog log = dto.toModel();
        String updateTime = rawSensorLogService.addLog(binId, log);
        return ResponseEntity.ok("Sensor log saved at " + updateTime);
    }

    /**
     * Lấy raw sensor logs gần nhất của 1 thùng rác.
     */
    @GetMapping("/bins/{binId}/sensor-logs")
    public ResponseEntity<List<BinRawSensorLog>> getRecentSensorLogs(
            @PathVariable String binId,
            @RequestParam(defaultValue = "30") int limit) {
        int safeLimit = Math.max(1, Math.min(limit, 200));
        return ResponseEntity.ok(rawSensorLogService.getRecentLogsForBin(binId, safeLimit));
    }
}
