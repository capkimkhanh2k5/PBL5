package com.iotSmartTrash.controller;

import com.iotSmartTrash.dto.BinRealtimeStatusResponseDTO;
import com.iotSmartTrash.dto.BinStatusUpdateDTO;
import com.iotSmartTrash.dto.RawSensorLogCreateDTO;
import com.iotSmartTrash.model.BinRawSensorLog;
import com.iotSmartTrash.model.BinRealtimeStatus;
import com.iotSmartTrash.service.BinRawSensorLogService;
import com.iotSmartTrash.service.BinRealtimeService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Controller cho IoT device (Raspberry Pi) gửi dữ liệu sensor.
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
     * Raspi gọi mỗi 30 giây để cập nhật trạng thái thùng rác.
     * Replaces RTDB write to: bins/{bin_id}
     */
    @PostMapping("/bins/{binId}/status")
    public ResponseEntity<String> updateBinStatus(
            @PathVariable String binId,
            @Valid @RequestBody BinStatusUpdateDTO dto) {
        BinRealtimeStatus status = dto.toModel();
        String updateTime = binRealtimeService.updateStatus(binId, status);
        return ResponseEntity.ok("Bin status updated at " + updateTime);
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
}
