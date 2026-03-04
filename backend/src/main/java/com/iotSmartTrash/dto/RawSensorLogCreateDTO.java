package com.iotSmartTrash.dto;

import com.iotSmartTrash.model.BinRawSensorLog;
import jakarta.validation.constraints.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO nhận raw sensor data từ Raspberry Pi.
 * Raspi gọi POST /api/v1/iot/bins/{binId}/sensor-logs mỗi 30 giây.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class RawSensorLogCreateDTO {

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

    /** epoch millis — nếu null thì server sẽ tự gán System.currentTimeMillis() */
    private Long recordedAt;

    public BinRawSensorLog toModel() {
        return BinRawSensorLog.builder()
                .temperature(this.temperature)
                .batteryLevel(this.batteryLevel)
                .fillOrganic(this.fillOrganic)
                .fillRecycle(this.fillRecycle)
                .fillNonRecycle(this.fillNonRecycle)
                .fillHazardous(this.fillHazardous)
                .recordedAt(this.recordedAt)
                .build();
    }
}
