package com.iotSmartTrash.dto;

import com.iotSmartTrash.model.BinSensorLog;
import com.iotSmartTrash.model.enums.BinPeriod;
import jakarta.validation.constraints.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class SensorLogCreateDTO {
    @NotBlank(message = "Bin ID is required")
    private String binId;

    @NotNull(message = "Period is required")
    private BinPeriod period;

    @NotBlank(message = "Date is required")
    private String date;

    private Double avgTemperature;
    private Double minTemperature;
    private Double maxTemperature;

    @Min(value = 0, message = "Battery >= 0")
    @Max(value = 100, message = "Battery <= 100")
    private Integer avgBattery;

    @Min(0)
    @Max(100)
    private Integer avgFillOrganic;
    @Min(0)
    @Max(100)
    private Integer avgFillRecycle;
    @Min(0)
    @Max(100)
    private Integer avgFillNonRecycle;
    @Min(0)
    @Max(100)
    private Integer avgFillHazardous;

    @Min(value = 1, message = "Sample count must be >= 1")
    private Integer sampleCount;

    public BinSensorLog toModel() {
        return BinSensorLog.builder()
                .binId(this.binId)
                .period(this.period)
                .date(this.date)
                .avgTemperature(this.avgTemperature)
                .minTemperature(this.minTemperature)
                .maxTemperature(this.maxTemperature)
                .avgBattery(this.avgBattery)
                .avgFillOrganic(this.avgFillOrganic)
                .avgFillRecycle(this.avgFillRecycle)
                .avgFillNonRecycle(this.avgFillNonRecycle)
                .avgFillHazardous(this.avgFillHazardous)
                .sampleCount(this.sampleCount)
                .build();
    }
}
