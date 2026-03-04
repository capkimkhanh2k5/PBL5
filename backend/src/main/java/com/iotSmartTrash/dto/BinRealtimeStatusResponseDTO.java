package com.iotSmartTrash.dto;

import com.iotSmartTrash.model.BinRealtimeStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO phản hồi trạng thái realtime của thùng rác.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BinRealtimeStatusResponseDTO {
    private String id;
    private String status;
    private Double temperature;
    private Integer batteryLevel;
    private Integer fillOrganic;
    private Integer fillRecycle;
    private Integer fillNonRecycle;
    private Integer fillHazardous;
    private Long lastUpdated;

    public static BinRealtimeStatusResponseDTO fromModel(BinRealtimeStatus model) {
        if (model == null) return null;
        return BinRealtimeStatusResponseDTO.builder()
                .id(model.getId())
                .status(model.getStatus())
                .temperature(model.getTemperature())
                .batteryLevel(model.getBatteryLevel())
                .fillOrganic(model.getFillOrganic())
                .fillRecycle(model.getFillRecycle())
                .fillNonRecycle(model.getFillNonRecycle())
                .fillHazardous(model.getFillHazardous())
                .lastUpdated(model.getLastUpdated())
                .build();
    }
}
