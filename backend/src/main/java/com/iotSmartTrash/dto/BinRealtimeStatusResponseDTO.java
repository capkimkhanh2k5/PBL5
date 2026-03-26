package com.iotSmartTrash.dto;

import com.iotSmartTrash.model.BinRawSensorLog;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO phản hồi trạng thái hiện tại của thùng rác.
 * Build trực tiếp từ BinRawSensorLog mới nhất.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BinRealtimeStatusResponseDTO {
    private String id;
    private String status;
    private Integer batteryLevel;
    private Integer fillOrganic;
    private Integer fillRecycle;
    private Integer fillNonRecycle;
    private Integer fillHazardous;
    private Long lastUpdated;

    /**
     * Tạo DTO từ raw sensor log mới nhất + tính status ONLINE/OFFLINE.
     */
    public static BinRealtimeStatusResponseDTO fromRawLog(String binId, BinRawSensorLog log, long offlineThresholdMs) {
        if (log == null) {
            return BinRealtimeStatusResponseDTO.builder()
                    .id(binId)
                    .status("UNKNOWN")
                    .batteryLevel(0)
                    .fillOrganic(0)
                    .fillRecycle(0)
                    .fillNonRecycle(0)
                    .fillHazardous(0)
                    .lastUpdated(0L)
                    .build();
        }

        long lastUpdated = log.getRecordedAt() != null ? log.getRecordedAt() : 0L;
        long ageMs = System.currentTimeMillis() - lastUpdated;
        String status = (lastUpdated > 0 && ageMs <= offlineThresholdMs) ? "ONLINE" : "OFFLINE";

        return BinRealtimeStatusResponseDTO.builder()
                .id(binId)
                .status(status)
                .batteryLevel(log.getBatteryLevel())
                .fillOrganic(log.getFillOrganic())
                .fillRecycle(log.getFillRecycle())
                .fillNonRecycle(log.getFillNonRecycle())
                .fillHazardous(log.getFillHazardous())
                .lastUpdated(lastUpdated)
                .build();
    }
}
