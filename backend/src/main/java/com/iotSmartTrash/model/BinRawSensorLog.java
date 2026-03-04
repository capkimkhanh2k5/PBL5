package com.iotSmartTrash.model;

import com.google.cloud.firestore.annotation.PropertyName;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Dữ liệu sensor thô từ Raspi
 *
 * Chỉ giữ 24 giờ gần nhất. Scheduler tự động xóa log cũ sau mỗi 6 tiếng.
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BinRawSensorLog {
    private String id; // auto-generated doc ID

    private Double temperature;

    @PropertyName("battery_level")
    private Integer batteryLevel;

    @PropertyName("fill_organic")
    private Integer fillOrganic;

    @PropertyName("fill_recycle")
    private Integer fillRecycle;

    @PropertyName("fill_non_recycle")
    private Integer fillNonRecycle;

    @PropertyName("fill_hazardous")
    private Integer fillHazardous;

    @PropertyName("recorded_at")
    private Long recordedAt; // epoch millis
}
