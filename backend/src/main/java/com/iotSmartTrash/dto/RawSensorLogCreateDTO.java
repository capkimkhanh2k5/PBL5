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
                .fillOrganic(this.fillOrganic)
                .fillRecycle(this.fillRecycle)
                .fillNonRecycle(this.fillNonRecycle)
                .fillHazardous(this.fillHazardous)
                .recordedAt(
                        this.recordedAt != null
                                ? com.google.cloud.Timestamp.ofTimeSecondsAndNanos(
                                this.recordedAt / 1000,
                                (int)((this.recordedAt % 1000) * 1_000_000)
                        )
                                : null
                )
                .build();
    }
}
