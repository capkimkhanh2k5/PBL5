package com.iotSmartTrash.dto;

import com.iotSmartTrash.model.BinRealtimeStatus;
import jakarta.validation.constraints.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO nhận dữ liệu trạng thái từ Raspberry Pi.
 * Raspi gửi mỗi 30 giây qua POST /api/v1/iot/bins/{binId}/status
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class BinStatusUpdateDTO {

    @NotBlank(message = "Status is required (ONLINE / OFFLINE / MAINTENANCE)")
    private String status;

    private Double temperature;

    @Min(value = 0, message = "Battery level >= 0")
    @Max(value = 100, message = "Battery level <= 100")
    private Integer batteryLevel;

    @Min(0) @Max(100)
    private Integer fillOrganic;

    @Min(0) @Max(100)
    private Integer fillRecycle;

    @Min(0) @Max(100)
    private Integer fillNonRecycle;

    @Min(0) @Max(100)
    private Integer fillHazardous;

    public BinRealtimeStatus toModel() {
        return BinRealtimeStatus.builder()
                .status(this.status)
                .temperature(this.temperature)
                .batteryLevel(this.batteryLevel)
                .fillOrganic(this.fillOrganic)
                .fillRecycle(this.fillRecycle)
                .fillNonRecycle(this.fillNonRecycle)
                .fillHazardous(this.fillHazardous)
                .build();
    }
}
