package com.iotSmartTrash.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BinPickupScheduleDTO {
    private String binId;
    private Integer avgFill;
    private String sourceDate;
    private String sourcePeriod;
    private Long sourceRecordedAt;
    private Long predictedPickupAt;
    private Integer predictedInHours;
    private String priority;
}
