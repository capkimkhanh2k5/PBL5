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
public class Alert {
    private String id;
    private String bin_id;
    private String alert_type; // VD: 'FULL_BIN', 'HIGH_TEMP', 'LOW_BATTERY'
    private String severity; // VD: 'CRITICAL', 'WARNING', 'INFO'
    private String message;
    private String status; // 'NEW', 'RESOLVED'
    private String resolved_by; // Tham chiếu đến UID người giải quyết
    private Timestamp created_at;
    private Timestamp resolved_at;
}
