package com.iotSmartTrash.model;

import com.google.cloud.Timestamp;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BinSensorLog {
    private String bin_id;
    private String period; // "00h", "06h", "12h", "18h"
    private String date; // "2024-02-17"
    private Double avg_temperature;
    private Double min_temperature;
    private Double max_temperature;
    private Integer avg_battery;
    private Integer avg_fill_organic;
    private Integer avg_fill_recycle;
    private Integer avg_fill_non_recycle;
    private Integer avg_fill_hazardous;
    private Integer sample_count;
    private Timestamp recorded_at;
}
