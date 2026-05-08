package com.iotSmartTrash.model;

import com.google.cloud.firestore.annotation.PropertyName;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import com.google.cloud.Timestamp;

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

    @PropertyName("battery_level")
    private Integer batteryLevel;

    @PropertyName("fillOrganic")
    private Integer fillOrganic;

    @PropertyName("fillRecycle")
    private Integer fillRecycle;

    @PropertyName("fillNonRecycle")
    private Integer fillNonRecycle;

    @PropertyName("fillHazardous")
    private Integer fillHazardous;

    @PropertyName("recordedAt")
    private Timestamp recordedAt; // epoch millis
}
