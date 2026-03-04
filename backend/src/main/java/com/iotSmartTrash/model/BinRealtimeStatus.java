package com.iotSmartTrash.model;

import com.google.cloud.firestore.annotation.PropertyName;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Trạng thái realtime của thùng rác 
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BinRealtimeStatus {
    private String id; // bin_id

    private String status; // ONLINE / OFFLINE / MAINTENANCE

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

    @PropertyName("last_updated")
    private Long lastUpdated; // epoch millis
}
