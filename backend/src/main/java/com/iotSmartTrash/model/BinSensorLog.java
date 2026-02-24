package com.iotSmartTrash.model;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.annotation.PropertyName;
import com.iotSmartTrash.model.enums.BinPeriod;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BinSensorLog {
    @PropertyName("bin_id")
    private String binId;

    private BinPeriod period;
    private String date;

    @PropertyName("avg_temperature")
    private Double avgTemperature;

    @PropertyName("min_temperature")
    private Double minTemperature;

    @PropertyName("max_temperature")
    private Double maxTemperature;

    @PropertyName("avg_battery")
    private Integer avgBattery;

    @PropertyName("avg_fill_organic")
    private Integer avgFillOrganic;

    @PropertyName("avg_fill_recycle")
    private Integer avgFillRecycle;

    @PropertyName("avg_fill_non_recycle")
    private Integer avgFillNonRecycle;

    @PropertyName("avg_fill_hazardous")
    private Integer avgFillHazardous;

    @PropertyName("sample_count")
    private Integer sampleCount;

    @PropertyName("recorded_at")
    private Timestamp recordedAt;
}
