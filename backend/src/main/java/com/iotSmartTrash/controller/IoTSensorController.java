package com.iotSmartTrash.controller;

import com.iotSmartTrash.dto.BinRealtimeStatusResponseDTO;
import com.iotSmartTrash.dto.RawSensorLogCreateDTO;
import com.iotSmartTrash.model.BinRawSensorLog;
import com.iotSmartTrash.service.BinRawSensorLogService;
import com.iotSmartTrash.service.BinRealtimeService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;

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
     * Lấy trạng thái hiện tại của 1 thùng rác (cho Flutter app).
     */
    @GetMapping("/bins/{binId}/status")
    public ResponseEntity<BinRealtimeStatusResponseDTO> getBinStatus(@PathVariable String binId) {
        return ResponseEntity.ok(binRealtimeService.getStatus(binId));
    }

    /**
     * Lấy trạng thái hiện tại của tất cả thùng rác (cho Flutter app).
     */
    @GetMapping("/bins/status")
    public ResponseEntity<List<BinRealtimeStatusResponseDTO>> getAllBinStatuses() {
        return ResponseEntity.ok(binRealtimeService.getAllStatuses());
    }

    // ━━━━━━━━━━━━━━━━━ Raw Sensor Logs ━━━━━━━━━━━━━━━━━

    /**
     * Raspi gọi mỗi 1h để ghi raw sensor data.
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
