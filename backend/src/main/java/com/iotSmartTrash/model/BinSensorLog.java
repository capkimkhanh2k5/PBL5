package com.iotSmartTrash.model;

import com.google.cloud.Timestamp;
import com.google.cloud.firestore.annotation.PropertyName;
import com.iotSmartTrash.model.enums.BinPeriod;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/*
    Model lưu trữ dữ liệu tổng hợp của Bin
    Dùng để lưu trữ dữ liệu làm phân tích, thống kê
*/
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BinSensorLog {
    @PropertyName("bin_id")
    private String binId;

    private BinPeriod period;

    private String date;

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
